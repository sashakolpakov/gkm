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
  point to call an unchanged named definition directly. The frozen GKM source
  audit admits 174 exact winning-source checkpoints, 149 exact adjacent
  transitions, 57 hard direct-call witnesses, and 14 sharp/direct-call
  intersections. These are a strict subset of the 181 replay-verified endpoint
  wins: `ft09` L2 and `tr87` L1--L6 are replay-valid deterministic
  reconstructions and are excluded from historical source-complexity
  denominators. OPINE has four hard witnesses and two sharp intersections;
  baseline1 has zero among its 18 exact adjacent transitions; Retrodict releases
  no executable winning entry point. The complete witness enumeration is
  generated in `arc/manuscript/generated/comparator_stats.md`.

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

The revision-bound manuscript target extracts the frozen source history and
recomputes GKM while reusing or rebuilding the independently pinned comparator
rows:

```bash
RELEASE_RECEIPT=../crack_lab/releases/arc_agi3_gkm_v2_181/receipts/140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681.json \
  make -C arc/manuscript reproduce
```

For a non-authoritative diagnostic against the mutable campaign tree:

```bash
python3 arc/audit_marginal_literal_reuse.py \
  --gkm-root arc/crack_lab/agent_solutions \
  --reuse-non-gkm-from-json arc/audit_results/marginal-literal-reuse.json \
  --retrodict-audit-json arc/audit_results/retrodict-solved-checkpoint-memory.json \
  --json /tmp/marginal-literal-reuse-current.json
```

That command is only a drift check; it is not the frozen manuscript authority.
Raw external-artifact commands and pinned input hashes
are documented in `arc/manuscript/opine_world_comparison.md`.

The source analyzers are:

- `arc/audit_gkm_solved_checkpoints.py`
- `arc/audit_baseline1_artifacts.py`
- `arc/audit_opine_solved_checkpoints.py`
- `arc/audit_retrodict_artifacts.py`
- `arc/audit_marginal_literal_reuse.py`
