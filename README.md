# GKM: Replay-Validated Self-Improvement for ARC-AGI-3

GKM is a verifier-driven program-growth loop for ARC-AGI-3 agents.
A proposer writes solver structure, the simulator verifies candidates by replay,
and marginal description length records what new structure was worth keeping.

The result is not only a solved-level count, but an audit trail of how competence
was acquired: promoted solver code, replay validation, WIP snapshots, charged
literals, reusable solver-leg refactors, and marginal-complexity accounting.

## Current replay-validated ARC-AGI-3 artifacts

| Game | Levels | Actions | Total marginal complexity | Artifact |
|---|---:|---:|---:|---|
| `wa30` | 9/9 | 596 | 1243 | `arc/crack_lab/agent_solutions/wa30_legs/` |
| `ls20` | 7/7 | 393 | 362 | `arc/crack_lab/agent_solutions/ls20_legs/` |

The central claim is that GKM converts proposer compute into replay-validated
solver structure. Reused legs are free, new reusable legs are charged once,
per-level glue is charged, and literal recovered paths are charged as literals.
The marginal-complexity trace therefore records when transfer is sufficient and
when the game forces new structure.

## Reproduce the ARC-AGI-3 artifacts

See [`REPRODUCE_ARC.md`](REPRODUCE_ARC.md).

The replay script is:

```bash
python arc/crack_lab/replay_scorecard.py --mode online
```

Artifact folders:

- [`wa30_legs`](arc/crack_lab/agent_solutions/wa30_legs/)
- [`ls20_legs`](arc/crack_lab/agent_solutions/ls20_legs/)

## Exact claim

The current public artifact claims replay-validated promoted solvers for:

- `wa30`: 9/9 levels, 596 actions, total marginal complexity 1243.
- `ls20`: 7/7 levels, 393 actions, total marginal complexity 362.

The claim is about promoted public artifacts and replay validation, not about
a private ARC-AGI-3 leaderboard result. The current repository is intended for
independent artifact review, reproduction, comparison, and extension.

---

## The broader GKM program: structure under free energy

Beyond the ARC-AGI-3 artifact, this repository collects small, controllable
substrates that test one idea from several angles: **free energy `F = R + λ·C`
used as a local selection principle** — where `R` is loss and `C` is a raw
description length — can drive the discovery, composition, and (we hope)
open-ended growth of structure.

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

Each domain guide is that domain's **hub**: it indexes the modules, runnable
experiments, tests, and reports that live alongside it, and links the domain's
self-contained LaTeX manuscript (`make -C <domain>/manuscript`):

| domain hub | manuscript | extras |
|---|---|---|
| [`foraging/FORAGING.md`](foraging/FORAGING.md) | [`foraging.tex`](foraging/manuscript/foraging.tex) | [open-ended-evolution thesis](foraging/OPEN_ENDED_EVOLUTION_THESIS.md) |
| [`transduction/TRANSDUCTION.md`](transduction/TRANSDUCTION.md) | [`transduction.tex`](transduction/manuscript/transduction.tex) | [benchmark report](transduction/register_transducer_benchmark.md) |
| [`bongard/BONGARD.md`](bongard/BONGARD.md) | [`free_energy_abstraction.tex`](bongard/manuscript/free_energy_abstraction.tex) | 5 reports linked in the hub |
| [`cone/CONE.md`](cone/CONE.md) | — (program doc: [`COLIMIT_CONE_APPROACH.md`](COLIMIT_CONE_APPROACH.md)) | 3 reports linked in the hub |
| [`arc/ARC.md`](arc/ARC.md) | [`arc_agi3.tex`](arc/manuscript/arc_agi3.tex) | [outreach one-pager](arc/manuscript/gkm_one_page_summary.md), promoted artifacts |

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
