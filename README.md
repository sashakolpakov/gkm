# GKM: structure under free energy

This repository collects small, controllable substrates that test one idea from
several angles: **free energy `F = R + λ·C` used as a local selection principle** —
where `R` is loss and `C` is a raw description length — can drive the discovery,
composition, and (we hope) open-ended growth of structure.

> Open-ended artificial evolution is possible under a free-energy paradigm if free
> energy is used as a local selection principle over agents embedded in an expanding,
> archive-driven ecology. Fixed-task free-energy minimization converges to
> compression; open-endedness requires that solved structures generate new validation
> pressures.

The shared mathematics — Kolmogorov complexity / MDL and the loss-complexity
(free-energy) lens of [arXiv:2507.13543](https://arxiv.org/abs/2507.13543) — is in
[`FREE_ENERGY_EXPLANATION.md`](FREE_ENERGY_EXPLANATION.md). Each domain applies the
same lens to a different substrate:

## Domains

- **[foraging/](foraging/FORAGING.md)** — open-ended evolution of sparse finite-state
  automata that play a visible grid-foraging game; free energy as the local selection
  rule, with lambda sweeps tracing a loss-complexity landscape. This is the founding
  thesis substrate ([`OPEN_ENDED_EVOLUTION_THESIS.md`](foraging/OPEN_ENDED_EVOLUTION_THESIS.md)).
- **[transduction/](transduction/TRANSDUCTION.md)** — synthesising compact
  deterministic **register transducers** from opaque-token pattern transitions;
  tiered primitives ask which capabilities a task family needs, with validation-frontier
  Pareto selection.
- **[bongard/](bongard/BONGARD.md)** — Bongard-style **concept induction** over
  opaque-object sequences, and the question of when free-energy accounting drives the
  **emergence of reusable abstraction** (encapsulated predicate macros) over duplicated
  rule bodies.
- **[cone/](cone/CONE.md)** — the substrate-agnostic core of the **colimit-cone
  program**: learn a compiled, verifiable *cone* over goal atoms rather than a
  monolithic policy; cone-leg discovery, goal induction from scalar reward, and the
  free-energy bound. The program document is
  [`COLIMIT_CONE_APPROACH.md`](COLIMIT_CONE_APPROACH.md).
- **[arc/](arc/ARC.md)** — the ARC-AGI-3 lift of the cone machinery (the offline
  connector, scene atoms, cone-leg discovery on games), plus a **self-improving
  agent** that cracks live ARC-AGI-3 keyboard games from the rawest interface,
  carrying only human preconceptions, with a single free-energy rule deciding what
  structure (a growing *leg library*) is kept. The agent lives in
  [`arc/crack_lab/`](arc/crack_lab/).

Each domain directory has its own one-page guide (linked above), its runnable
experiments, its tests, and — for foraging, transduction, and bongard — a
self-contained LaTeX manuscript under `<domain>/manuscript/` (`make -C <domain>/manuscript`);
the ARC manuscript is [`arc/manuscript/arc_agi3.tex`](arc/manuscript/arc_agi3.tex).

## Tests

Every domain's tests run from the repository root (a top-level `conftest.py` puts each
domain directory on the path):

```bash
python -m pytest foraging/test_evo_game.py transduction/test_pattern_fsa.py \
    bongard/test_bongard_sparse_classifier.py bongard/test_abstraction_emergence.py \
    cone/ arc/test_arc_agi3_adapter.py arc/test_arc_goal_induction.py \
    arc/test_arc_scene_atoms.py arc/test_cone_leg_discovery.py -q
```

## Documentation

The Sphinx documentation source is in [`docs/`](docs/) and deploys through the included
GitHub Pages workflow (<https://sashakolpakov.github.io/gkm/>):

```bash
python3 -m sphinx -W -b html docs docs/_build/html
```
