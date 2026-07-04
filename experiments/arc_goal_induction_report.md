# Goal Induction on a Synthetic ARC-Shaped Fixture

> SCOPE / HONESTY: this experiment runs on the **synthetic `GoalGame` stub**, a
> reproducible coloured-grid fixture that mimics the ARC interface shape — it is
> NOT real ARC data. It validates the goal-induction *machinery* deterministically
> and offline. For the real ARC API (live frames + scene functor on genuine
> games), see `experiments/arc_live_report.md`. The two are complementary: this
> fixture proves the algorithm; the live probe proves the perception runs on
> real frames.

This is the lift the connector needed (COLIMIT_CONE_APPROACH.md R12 gap b,
Section 13.4): the foraging goal-induction loop, run unchanged in form, over
the connector's scene features and a hidden game score. The agent does not
know the game's objective; it infers which colour and which relation the score
tracks, then compiles the inferred goal into a cone of colour-bound seek/flee
phases. The substrate here is the synthetic fixture; the same loop is meant to
run on real frames once the live action round-trip is settled (see the live
report).

## Reproduction

```bash
python3 experiments/run_arc_goal_induction.py
python3 -m unittest tests.test_arc_goal_induction
```

Outputs to `output/arc_goal_induction/summary.json`. Modules:
`arc_goal_induction.py` (the lift), `arc_agi3_adapter.py` (scene functor,
colour-slot observation), `cone_goal_induction.py` (the reused free-energy
induction core).

## The lift, term by term

```text
foraging (cone_goal_induction)        ARC (arc_goal_induction)
------------------------------        -----------------------------------------
channel FOOD/HOME/HAZARD              colour slot (an object colour)
feature food/home/safe               predicate@colour atom: clear@c, avoid@c
hidden task loss                     hidden game score (scalar)
seek / flee leg (channel-blind)      SAME legs, bound to a colour slot
outcome_features(episode)            scene_atoms(game) over candidate colours
compile_goal -> bound cone           goal_to_cone -> seek/flee colour phases
induce_active (disagreement)         induce_arc_goal (disagreement, same core)
```

The free-energy induction core (`goal_free_energy`, `induce_goal`,
`discrimination`, `candidate_goals`) is literally reused — it was generalized
to accept an arbitrary feature-key set, so the same code selects food/home/safe
subsets for foraging and predicate@colour subsets for ARC.

Goal atoms:

```text
clear@c   collect (remove on contact) all objects of colour c   -> seek leg
avoid@c   end >= SAFE_RADIUS from every colour-c object          -> flee leg
```

The agent sees the atoms and the scalar score; it never reads the objective or
which colours are collectible. With 3 candidate colours the atom vocabulary has
6 atoms; the agent must find which 1-2 the score depends on.

## Result (candidate colours {2,3,5}, 6 probe instances, 6 held-out, lambda 0.05)

```text
hidden_objective,inferred_goal,match,rounds,probe_episodes,holdout_solved
clear_2,clear@2,True,2,12,1.00
avoid_5,avoid@5,True,1,6,1.00
clear_2_3,clear@2+clear@3,True,4,24,1.00
clear_2_avoid_5,clear@2+avoid@5,True,4,24,1.00
clear_3_avoid_5,clear@3+avoid@5,True,4,24,1.00
```

All five hidden objectives are induced exactly, and every compiled cone solves
all held-out game instances. Single-atom objectives take 1-2 probe rounds;
two-atom objectives take 4 (the disagreement loop must run a pair probe to see
the score reach 1.0, since neither single behavior satisfies a two-atom goal).

## Observations

1. **The loop lifts without changing the induction core.** Only the substrate
   adapters changed (scene atoms instead of foraging features, colour-bound
   phases instead of channel-bound calls, game score instead of task loss).
   The free-energy model selection is the same function. This is the concrete
   payoff of the Section 0 framing: goal induction is substrate-independent.

2. **The agent infers both colour and relation.** With six candidate atoms it
   recovers, e.g., `clear@2` and rejects `clear@3` (a distractor colour
   present in the scene) and `avoid@2`. This is the ARC-faithful version of
   "what I do to red vs blue is different unless I build the abstraction": the
   colour is part of what must be induced, not given.

3. **Distractor colours are correctly excluded.** Colour 3 is present in the
   `clear_2` games but never collectible and never in the objective; its atoms
   stay constant across probes and free energy drops them. (Tested directly.)

4. **Disagreement-driven exploration sets the probe count.** Single-atom goals
   converge as soon as one informative probe opens a free-energy margin;
   two-atom goals need a pair probe (collect-then-avoid) to observe the score
   reaching 1.0, which the loop selects because that probe maximally separates
   the surviving candidate goals.

5. **Execution is witness-leg limited, not induction limited.** On cramped
   grids the wall-blind witness flee leg occasionally corners and fails to
   reach safety; the grid is sized (14x14) so pure antipodal flight has escape
   room, the same rationale as the foraging hazard placement. Induction is
   correct regardless of execution; an evolved flee leg with wall sense would
   remove the grid-size dependence.

## How this connects to a live ARC-AGI-3 game

The stub `GoalGame` stands in for a real environment: it exposes only a scalar
score and a frame, removes objects on contact, and hides its objective — the
same interface shape as the live API (`arc_agi3_adapter.ArcEnv`). To run on a
live game, three things must be supplied that the stub provides for free:

1. **The atom vocabulary.** Here `clear@c` / `avoid@c` are hand-defined, as the
   foraging features were. A live game needs a richer, discovered scene-atom
   vocabulary (object counts, containment, adjacency, frame-to-frame deltas).
   Feature *discovery* from raw frames is the standing open problem, inherited
   by the scene functor, not solved here.
2. **The avatar identification.** The scene functor's avatar heuristic
   (declared colour, or smallest singleton) must hold or be learned.
3. **`ACTION6` and richer dynamics.** Coordinate actions and hidden state
   transitions are out of scope for this first lift.

What is demonstrated: given a scene-atom vocabulary and a scalar score, the
agent infers the hidden goal by free energy and acts through a cone — the
complete loop, end to end, on the ARC-shaped interface. What remains is the
perception frontier (atom discovery) and the live-transport details (R12 gap
a), not the induction loop itself.

## Caveats

- Atoms are hand-defined (the standing feature-discovery limitation).
- Single seed family per objective (deterministic given seeds); a broader seed
  sweep and a lambda-parsimony diagnostic (as in the foraging report) are the
  natural hardening steps.
- Witness legs; evolved legs would remove the grid-size dependence and tighten
  paths.
- The stub game is a fixture, not a real ARC game; it shares the interface
  shape, not the difficulty.

## Next steps

- A richer, partly-discovered scene-atom vocabulary so the colour/relation
  space is not fully enumerated by hand.
- Evolved seek/flee legs shared with the foraging substrates.
- A lambda-parsimony diagnostic for objectives whose atoms are correlated by
  the game geometry (the ARC analogue of foraging's flee_then_home).
- A single authenticated round-trip against the live API to confirm transport
  (R12 gap a), then the same loop on a real game's score.
