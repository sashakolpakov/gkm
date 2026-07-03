# Scene-Atom Discovery from Raw Frames (synthetic fixture)

> SCOPE / HONESTY: "raw frames" here are the **synthetic `GoalGame` stub's**
> frames, not real ARC frames. This validates the discovery *machinery*
> (avatar/colour/atom discovery, variance pruning) deterministically and
> offline. The real-ARC counterpart — the scene functor run on genuine live
> frames — is in `experiments/arc_live_report.md`, where the connected-component
> object decomposition is confirmed to run on real ls20/wa30 frames. The avatar
> discovery primitive's transfer to real frames awaits the live action loop.

The previous goal-induction step (experiments/arc_goal_induction_report.md)
was handed its atom vocabulary: the candidate colours and the clear/avoid
evaluators were supplied. This step removes that hand-holding. The agent
discovers the avatar, the object colours, and the live atoms from frames,
then runs goal induction over the DISCOVERED vocabulary.

## Reproduction

```bash
python3 experiments/run_arc_goal_induction.py                     # discovered vocabulary (default)
python3 experiments/run_arc_goal_induction.py --oracle-vocabulary  # hand-given atoms, for comparison
python3 -m unittest tests.test_arc_scene_atoms
```

Outputs to `output/arc_goal_induction/summary.json`. Module:
`arc_scene_atoms.py`; it feeds `arc_goal_induction.discover_and_induce`, which
reuses the same free-energy induction core.

## What is discovered, and what is still given

```text
discovered from raw frames                  still given (legitimate prior)
------------------------------------------  ------------------------------------------
the avatar: which object the actions move    the action set (ACTION1-5)
the object colours present                   the relation SCHEMAS {clear, reach, avoid}
which (colour, relation) atoms are live       (the morphism vocabulary, Section 0)
  (by variance pruning under exploration)
```

The avatar is found by **action response**: from a fresh frame, each
directional action is issued and the colour whose object translates by the
action's delta is the avatar (votes across the four actions). This is the
first cone any ARC agent must learn — "which object do my actions control?" —
and it is not hardcoded (a regression test discovers a non-standard avatar
colour 7).

The atom vocabulary is **template instantiation + variance pruning**: the
relation schemas are applied to each discovered colour, then atoms whose value
does not vary under the exploration probes are dropped. A constant atom carries
no information; pruning it is the discovery of which (colour, relation) pairs
are live. This is the susceptibility / informativeness idiom used elsewhere in
the repository, here selecting features rather than diagnosing phase
transitions.

## Result (candidate colours {2,3,5}, 6 probe instances, 6 held-out, lambda 0.05)

```text
hidden_objective,avatar,discovered_colors,atoms_kept,atoms_pruned,inferred_goal,name_match,cone_match,rounds,probe_episodes,holdout_solved
clear_2,4,2|3|5,7,2,clear@2,True,True,2,12,1.00
avoid_5,4,2|3|5,6,3,avoid@5,True,True,2,12,1.00
clear_2_3,4,2|3|5,5,4,clear@2+clear@3,True,True,3,18,1.00
clear_2_avoid_5,4,2|3|5,7,2,clear@2+avoid@5,True,True,3,18,1.00
clear_3_avoid_5,4,2|3|5,8,1,clear@3+avoid@5,True,True,2,12,1.00
```

(9 atoms are instantiated = 3 schemas x 3 colours; `atoms_kept + atoms_pruned`
= 9 in every row.)

## Observations

1. **The avatar is discovered, not assumed.** Every row found avatar colour 4
   by action response, and the pipeline never reads the game's `avatar_color`.
   A separate test discovers colour 7 when that is the moving object, so the
   mechanism is genuinely "what do my actions move", not a constant.

2. **Colours and live atoms are discovered from frames.** The candidate
   colours {2,3,5} are read off the frame content; the 9 instantiated atoms are
   pruned by observed variance to between 5 and 8 live atoms, the pruned set
   being the uninformative (constant) atoms for that objective's dynamics.

3. **Induction over the discovered vocabulary recovers the exact goal.** All
   five objectives are recovered with both name match (the inferred atom set
   equals the objective) and cone match (the compiled phases equal the
   ground-truth cone), and every compiled cone solves all held-out instances.
   Discovery did not cost correctness relative to the oracle vocabulary.

4. **Variance pruning is data-dependent and honest.** Different objectives
   prune different atoms (1-4 pruned), because which colours are collectible —
   and therefore which atoms vary under exploration — depends on the hidden
   dynamics the agent is probing. The pruning reflects what the agent actually
   observed, not a fixed mask.

5. **Pruning is an efficiency/interpretability win, not a correctness
   crutch.** Free energy would tolerate the extra constant atoms (they never
   predict the score), so induction succeeds with or without pruning; pruning
   shrinks the candidate-goal enumeration and yields an interpretable "these
   (colour, relation) pairs are live" report. The `--oracle-vocabulary` flag
   runs the hand-given path for direct comparison.

## What this closes and what remains

This closes the headline half of the perception frontier flagged in R12/R15:
the agent no longer needs the atom vocabulary handed to it for the ARC-shaped
interface. The avatar, the object colours, and the live (colour, relation)
atoms are all discovered from raw frames; only the relation SCHEMA library
{clear, reach, avoid} remains a given — and that is the legitimate inductive
bias of the morphism vocabulary (Section 0, consequence 1: what can be learned
is fixed by the morphisms).

Remaining frontiers:

- **Schema discovery.** The relation schemas are still hand-written. Inducing
  the schemas themselves (e.g. containment, alignment, count-equality) from
  frame deltas is the deeper open problem; the `SceneDelta` vocabulary in the
  connector (moved/appeared/vanished) is the natural starting point.
- **Richer scenes.** Multi-cell objects, several avatars, and non-grid-aligned
  layouts stress the centroid/component heuristics.
- **Live transport.** The same R12 gap (a): one authenticated round-trip to
  confirm the API header and payload, then this pipeline on a real game's
  frames and score.

## Caveats

- The relation schemas are hand-defined (the next frontier, stated above).
- The stub `GoalGame` is a reproducible fixture, not a real ARC game; it shares
  the interface shape and the discovery challenges (avatar, colours, dynamics),
  not the visual difficulty.
- Execution remains witness-leg limited on cramped grids (wall-blind flee);
  the grid is sized for escape room, and evolved legs are the fix.
- Single seed family per objective; a broader sweep and a variance-threshold
  sensitivity check are natural hardening steps.
