# Foraging Under Free-Energy Selection

This subject studies sparse finite-state agents in a visible grid-foraging ecology.
It is the repository's controlled setting for measuring the tradeoff between episode
loss and encoded policy size under `F = R + lambda * C`.

Unlike the ARC source-growth ledger, the foraging complexity term prices an explicit
automaton representation. Lambda sweeps can therefore trace a finite empirical
loss-complexity frontier. The current experiments are a substrate study; they do not
establish open-ended evolution in unrestricted environments.

## Entry Points

- [`FORAGING.md`](FORAGING.md): full subject guide.
- [`OPEN_ENDED_EVOLUTION_THESIS.md`](OPEN_ENDED_EVOLUTION_THESIS.md): research thesis
  and falsification conditions.
- [`evo_game.py`](evo_game.py): sparse automaton substrate.
- [`run_foraging_ecology.py`](run_foraging_ecology.py): experiment runner and lambda
  sweeps.
- [`manuscript/foraging.tex`](manuscript/foraging.tex): subject manuscript.

Run from the repository root:

```bash
python3 foraging/run_foraging_ecology.py --generations 80 --population 160
python -m pytest foraging/test_evo_game.py -q
```
