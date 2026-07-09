# Goal Induction in Foraging

The missing core (COLIMIT_CONE_APPROACH.md R12 gap 2): in every prior
experiment the agent is *told* its task through the loss. Real ARC-AGI-3 games
hide the goal and expose only a scalar score. This experiment builds goal
induction in the familiar foraging context — the agent infers what the reward
rewards, then compiles that goal into a cone — designed so the same loop lifts
to the ARC connector.

## Reproduction

```bash
python3 cone/run_cone_goal_induction.py
python3 -m unittest tests.test_cone_goal_induction
```

Outputs to `output/cone_goal_induction/summary.json`. Module:
`cone_goal_induction.py`; it compiles inferred goals into the v3 bound
substrate (`cone_foraging_bound.py`), so bindings are priced.

## The reduction

A hidden task rewards the agent for the mean satisfaction of a hidden subset
of three OBSERVABLE outcome features — food collected, at home, safe from
hazard. The agent sees the features and the scalar reward, never which subset
the reward averages, and never the task name. Goal induction is then
free-energy model selection over feature subsets:

```text
F(S) = misfit( predict reward as mean of S's features, across probes ) + lambda * |S|
```

This is the same MDL feature selection the Bongard and abstraction experiments
run, now over GOAL features learned from interaction rather than labels. The
loop:

1. Run probe cones (built from the leg library: seek/flee bound to channels)
   and record (outcome features, scalar reward) per probe episode.
2. Disagreement-driven exploration: each round, run the untested probe cone
   whose outcomes most separate the current top candidate goals; stop when the
   best goal's free-energy margin over the runner-up is comfortable.
3. Induce the minimal-free-energy goal; compile it to a bound cone (one phase
   per feature, seek for food/home, flee for hazard); verify on held-out
   levels.

## Result (seed 29, lambda 0.05, 6 probe levels, 6 held-out levels)

```text
hidden_task,true_goal,inferred_goal,match,rounds,probe_episodes,holdout_solved
forage,food,food,True,2,12,1.00
homing,home,home,True,1,6,1.00
forage_then_home,food+home,food+home,True,2,12,1.00
flee,safe,safe,True,1,6,1.00
forage_flee,food+safe,food+safe,True,2,12,1.00
flee_then_home,home+safe,home,False,7,42,1.00
```

Parsimony diagnostic on the one confounded task:

```text
lambda,inferred_goal,holdout_solved
0.005,home+safe,1.00
0.020,home,1.00
0.050,home,1.00
```

## Observations

1. **Five of six hidden goals are induced exactly, from very few
   interactions.** Single-feature goals (forage, homing, flee) take one probe
   round (6 episodes); two-feature goals (forage_then_home, forage_flee) take
   two rounds (12 episodes). The compiled cone solves every held-out level.
   The agent recovered the goal without ever being told the task.

2. **The sixth case is a correct parsimony result, not a failure.** In
   `flee_then_home` the home cell is generated outside the hazard radius
   (substrate constraint), so reaching home implies safety: the features
   `home` and `safe` are nearly perfectly correlated on any cone that reaches
   home, and the reward `(home+safe)/2` is almost a function of `home` alone.
   Free energy therefore prefers the simpler goal `home`, which both explains
   the reward (misfit gap ~0.012) and solves the task (held-out 1.00). The
   λ-sweep confirms this is parsimony, not error: at λ=0.005 the agent
   recovers the full `home+safe`; at λ≥0.02 the complexity penalty outweighs
   the tiny misfit improvement and it returns `home`. This is the structure
   function of goal induction — which goal you infer depends on parsimony
   pressure, and the extra feature is dropped precisely when the environment
   does not reward it distinguishably.

3. **Disagreement-driven exploration pays its way.** Single-feature goals
   converge in one round because the first informative probe opens a
   free-energy margin immediately. The confounded task exhausts the budget (7
   rounds) precisely because no margin ever opens between `home` and
   `home+safe` — the honest signature of an unidentifiable distinction, not
   wasted search.

4. **Goal induction reuses the existing free-energy machinery unchanged.**
   The inducer is MDL feature selection (misfit + λ·complexity), identical in
   form to the Bongard/abstraction selectors; only the data source changed
   (interaction reward instead of labels) and the output is compiled to a cone
   instead of a classifier.

## ARC-AGI-3 correspondence

This is the loop the ARC connector needs. There the observable features become
scene predicates (object counts, containment, deltas), the scalar reward is
the game score, and the compiled cone issues `CALL(leg, colour-slot)` actions.
The induction principle is identical: infer the simplest goal predicate whose
satisfaction tracks the reward, under free energy, then act through a cone that
achieves it. The `flee_then_home` parsimony case is the foraging shadow of a
general ARC risk: when two goal hypotheses are reward-equivalent under the
observed interactions, the agent commits to the simpler one and only a
distinguishing situation (or lower λ) separates them.

## Caveats

- The observable feature vocabulary (food/home/safe) is hand-defined here, as
  the Bongard primitive atoms are; goal induction discovers which features the
  reward tracks, not the features themselves. Discovering the feature
  vocabulary from raw scenes is the same open problem flagged throughout the
  thesis, now inherited by the ARC connector's scene functor.
- The probe cones are built from witness legs; an evolved-leg version is the
  natural hardening step (and shared with the v3 report's next steps).
- Single seed. The accounting/induction is deterministic given the probe
  levels; seed-robustness of the λ-thresholds for the confounded task is worth
  a sweep.

## Next steps

- Lift goal induction onto the ARC connector's scene features and the stub
  game's hidden score, then a live game once a key is available.
- Replace witness legs with evolved/lifted legs so the whole pipeline —
  perceive, induce goal, compile cone, act — is search-grounded end to end.
- Generalize beyond conjunctive goals (ordered goals, conditional goals) and
  test whether free energy still recovers the minimal correct structure.
