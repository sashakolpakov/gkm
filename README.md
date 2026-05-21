# GKM: Open-Ended Evolution Under Free Energy

This repository is focused on a research thesis and a small automata-evolution substrate.

The thesis:

> Open-ended artificial evolution is possible under a free-energy paradigm if free energy is used as a local selection principle over agents embedded in an expanding, archive-driven ecology. Fixed-task free-energy minimization converges to compression; open-endedness requires that solved structures generate new validation pressures.

See [OPEN_ENDED_EVOLUTION_THESIS.md](OPEN_ENDED_EVOLUTION_THESIS.md) for the full argument.

## Current Research Substrate

The active experiment is a small finite-state automaton ecology:

- Agents are table-driven automata.
- Genomes mutate by explicit rule edits.
- Agents play a visible grid foraging game.
- Selection uses an explicit free-energy objective:

```text
F_lambda(a) = R(a) + lambda C(a)
```

where:

- `R(a)` is the loss function: missed resources plus small step and collision costs.
- `C(a)` is the selected complexity function. The runner can optimize active, table, pruned, or mixed complexity.
- `lambda` controls pressure toward compact policies.

Complexity modes:

- `active`: states and rules actually used in observed episodes.
- `table`: the whole encoded transition table, so unused capacity is still paid for.
- `pruned`: all transition rules reachable from state 0 under any possible observation.
- `mixed`: active complexity plus a dead-code tax from the unused table.

The runner sweeps `lambda` to trace a loss-complexity landscape, following the structure-function/free-energy viewpoint in [arXiv:2507.13543](https://arxiv.org/abs/2507.13543).

This is not intended as a final open-ended system. It is a controllable instrument for studying the first necessary pieces: heritable structure, visible behavior, complexity pressure, replayable lineages, and eventually frontier-generating environments.

## Run

```bash
python agent.py --generations 80 --population 160 --render
```

Equivalent:

```bash
python evo_game.py --generations 80 --population 160 --render
```

Use Hyperopt/TPE instead of the genetic population loop:

```bash
pip install -r requirements.txt
python agent.py --optimizer hyperopt --hyperopt-evals 300 --lambda-points 5
```

Compare complexity assumptions:

```bash
python agent.py --complexity-mode active --lambda-points 5
python agent.py --complexity-mode table --lambda-points 5
python agent.py --complexity-mode pruned --lambda-points 5
python agent.py --complexity-mode mixed --lambda-points 5
```

Outputs are written to `output/evo_game/`:

```text
best_automaton.py        exported evolved policy
evolution_history.json   generation metrics
lambda_sweep.json        per-lambda loss/complexity/free-energy records, including all complexity metrics
summary.json             train/validation summary
best_replay.txt          ASCII replay of the final best policy
```

## Research Direction

The intended direction is:

```text
automata -> interaction -> archive -> frontier environments -> lambda sweeps -> lineage analysis
```

The goal is not to make a clever game bot. The goal is to test whether free-energy selection can support continued structural innovation when the ecology itself expands.

## Near-Term Experiments

1. **Closed-world baseline**
   Evolve automata on fixed maps. Prediction: rapid improvement, then stagnation/compression.

2. **Frontier curriculum**
   Generate new maps near the competence boundary. Prediction: solve-expand-solve cycles.

3. **Lambda phase diagram**
   Sweep `lambda` and measure different loss-complexity frontiers. `lambda_sweep.json` includes `complexity_variance`, a susceptibility-style diagnostic for phase-transition candidates.

4. **Archive validation**
   Keep environments that distinguish lineages. Prediction: multiple strategies persist.

5. **Coevolution**
   Add other agents, resource depletion, markers, or adversaries. Prediction: new niches create new selection pressures.

## Files

```text
agent.py                         compatibility entry point for evo_game.py
evo_game.py                      finite-state automata evolution experiment
OPEN_ENDED_EVOLUTION_THESIS.md   thesis and experimental program
FREE_ENERGY_EXPLANATION.md       mathematical background
tests/test_evo_game.py           standard-library tests
requirements.txt                 optional Hyperopt/TPE dependency
```

## Tests

```bash
python -m unittest
python -m py_compile agent.py evo_game.py tests/test_evo_game.py
```
