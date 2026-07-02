# Foraging: open-ended evolution under free energy

This domain studies the founding GKM question in its most controllable form: can
**free energy used as a local selection rule** support continued structural
innovation, rather than collapsing to a single compressed solution? Agents are
sparse deterministic finite-state automata that play a visible grid-foraging game;
selection minimises `F = R + λ·C` over an evolving population.

- **Thesis:** [`OPEN_ENDED_EVOLUTION_THESIS.md`](OPEN_ENDED_EVOLUTION_THESIS.md) —
  the full argument that fixed-task free-energy minimisation converges to
  compression, and open-endedness requires solved structure to generate new
  validation pressure.
- **Common math:** [`../FREE_ENERGY_EXPLANATION.md`](../FREE_ENERGY_EXPLANATION.md).
- **Manuscript:** [`manuscript/foraging.tex`](manuscript/foraging.tex)
  (build with `make -C manuscript`).

## The substrate

An agent is a sparse FSA with an explicitly encoded transition relation. The input
is relational, not raw:

```text
(current_state, previous_move, relative_food_azimuth) -> (move_sequence, next_state)
```

A rule may carry a short *sequence* of moves (a macro), not just one. If no encoded
rule matches an input the automaton **halts** the episode — there is no random
fallback and no free default move. Genomes mutate by explicit rule edits,
additions, and deletions, so structure is heritable and inspectable.

## Free energy

Selection is the free-energy objective

```text
F_lambda(a) = R(a) + lambda * C(a)
```

- `R(a)` — loss: missed resources plus small step and collision costs.
- `C(a)` — raw description length of the genome. Four complexity modes let us
  compare accounting assumptions:
  - `table` (default): sum over the whole encoded sparse rule set — extra rules and
    long macros are paid for;
  - `active`: only rules exercised in observed episodes;
  - `pruned`: rules reachable from state 0 under any observation;
  - `mixed`: active complexity plus a dead-code tax.
- `lambda` sets the pressure toward compact policies. Sweeping `lambda` traces a
  loss-complexity (free-energy) landscape in the sense of
  [arXiv:2507.13543](https://arxiv.org/abs/2507.13543).

`lambda_sweep.json` records `complexity_variance`, a susceptibility-style diagnostic
for phase-transition candidates along the sweep.

## Key modules

- `evo_game.py` — the grid-foraging FSA substrate library (importable, silent).
- `run_foraging_ecology.py` — the experiment runner (lambda sweeps, complexity
  modes, optional Hyperopt/TPE optimiser, ASCII replay).
- `agent.py` — a thin compatibility entry point (`from run_foraging_ecology import main`).

## Run

```bash
python3 foraging/run_foraging_ecology.py --generations 80 --population 160 --render
# compatibility entry point:
python3 foraging/agent.py --generations 80 --population 160 --render
# compare complexity assumptions:
python3 foraging/run_foraging_ecology.py --complexity-mode active --lambda-points 5
python3 foraging/run_foraging_ecology.py --complexity-mode table  --lambda-points 5
# TPE instead of the genetic loop (needs requirements.txt):
python3 foraging/run_foraging_ecology.py --optimizer hyperopt --hyperopt-evals 300 --lambda-points 5
```

Outputs land in `output/evo_game/` (`best_automaton.py`, `evolution_history.json`,
`lambda_sweep.json`, `summary.json`, `best_replay.txt`).

## Tests

```bash
python -m pytest foraging/test_evo_game.py -q
```

## Where this is headed

```text
automata -> interaction -> archive -> frontier environments -> lambda sweeps -> lineage analysis
```

The near-term program: a closed-world baseline (expect improvement then
compression/stagnation), a frontier curriculum near the competence boundary (expect
solve–expand–solve cycles), a lambda phase diagram, archive validation, and
coevolution. The goal is not a clever game bot; it is to test whether free-energy
selection can sustain structural innovation when the ecology itself expands.
