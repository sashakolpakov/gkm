# GKM: Auditable Program-Growth Experiments

GKM is a collection of verifier-driven program-growth experiments. In the
ARC-AGI-3 study, a proposer writes solver code and the simulator validates promoted
behavior by fresh replay. A source-size ledger records positive net growth of the
retained library and player files.

The result is not only a solved-level count, but an audit trail of how competence
was acquired: promoted solver code, replay validation, WIP snapshots, charged
literals, reusable solver-leg refactors, and marginal-complexity accounting.

## Current replay-validated ARC-AGI-3 artifacts

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The definitive [Competition-Mode scorecard](https://arcprize.org/scorecards/9e166671-0953-42f3-89de-a0fd57d7b147)
scores **17.136507936507936%** across all 25 public games: 37/183 levels in
1456 API actions including resets. One unchanged game-agnostic architecture
completes `wa30` and `ls20`, reaches L4 on `ft09`, `g50t`, `r11l`, `sp80`, and
`tr87`, and reaches L1 on `tu93`.

The endpoint claims are replay claims: the action counts are the final validated
paths, not totals for proposal, search, or cloned lookahead. The historical growth
charge is computed as the positive net change in each of two files, with an AST
surcharge for container literals. Unchanged legs incur no new charge, but additions
and deletions within one file can cancel. Consequently, a low value is evidence of
reuse only when the source diff and replay also show reuse; the scalar alone is not a
semantic novelty detector.

## Reproduce the ARC-AGI-3 artifacts

See [`REPRODUCE_ARC.md`](REPRODUCE_ARC.md).

The replay script is:

```bash
python arc/crack_lab/replay_scorecard.py --mode online
```

The definitive all-game card used:

```bash
python arc/crack_lab/replay_scorecard.py --mode competition \
  --games wa30,ls20,ft09,g50t,r11l,sp80,tr87,tu93
```

Artifact folders:

- [`wa30_legs`](arc/crack_lab/agent_solutions/wa30_legs/)
- [`ls20_legs`](arc/crack_lab/agent_solutions/ls20_legs/)

## Exact claim

The current public artifact claims replay-validated promoted solvers for:

- `wa30`: 9/9 levels, a 596-action validated replay, and the complete published
  ledger `112, 78, 95, 47, 405, 225, 145, 204, 147`, totaling 1458. Its unchanged
  operational checkpoint retains only the records accumulated after its resume base.
- `ls20`: 7/7 levels and a 393-action validated replay, with a complete seven-entry
  growth ledger totaling 362.

The local harness exposes state cloning for lookahead. The official ARC-AGI-3
environment wrapper exposes `reset()` and `step()`, not arbitrary state forking.
The scorecard is an official replay score, but it does not measure discovery
interaction efficiency, sample efficiency, or compute-matched proposer performance.
The repository supplies the corresponding artifacts for review, reproduction, and
extension.

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

- **[foraging/](foraging/README.md)** — open-ended evolution of sparse finite-state
  automata that play a visible grid-foraging game; free energy as the local selection
  rule, with lambda sweeps tracing a loss-complexity landscape. This is the founding
  thesis substrate ([`OPEN_ENDED_EVOLUTION_THESIS.md`](foraging/OPEN_ENDED_EVOLUTION_THESIS.md)).
- **[transduction/](transduction/README.md)** — synthesising compact
  deterministic **register transducers** from opaque-token pattern transitions;
  tiered primitives ask which capabilities a task family needs, with validation-frontier
  Pareto selection.
- **[bongard/](bongard/README.md)** — Bongard-style **concept induction** over
  opaque-object sequences, and the question of when free-energy accounting drives the
  **emergence of reusable abstraction** (encapsulated predicate macros) over duplicated
  rule bodies.
- **[cone/](cone/README.md)** — the substrate-agnostic core of the **colimit-cone
  program**: learn a compiled, verifiable *cone* over goal atoms rather than a
  monolithic policy; cone-leg discovery, goal induction from scalar reward, and the
  free-energy bound. The program document is
  [`COLIMIT_CONE_APPROACH.md`](COLIMIT_CONE_APPROACH.md).
- **[arc/](arc/README.md)** — the ARC-AGI-3 lift of the cone machinery (the offline
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
| [`foraging/README.md`](foraging/README.md) | [`foraging.tex`](foraging/manuscript/foraging.tex) | [open-ended-evolution thesis](foraging/OPEN_ENDED_EVOLUTION_THESIS.md) |
| [`transduction/README.md`](transduction/README.md) | [`transduction.tex`](transduction/manuscript/transduction.tex) | [benchmark report](transduction/register_transducer_benchmark.md) |
| [`bongard/README.md`](bongard/README.md) | [`free_energy_abstraction.tex`](bongard/manuscript/free_energy_abstraction.tex) | reports linked in the hub |
| [`cone/README.md`](cone/README.md) | — (program doc: [`COLIMIT_CONE_APPROACH.md`](COLIMIT_CONE_APPROACH.md)) | 3 reports linked in the hub |
| [`arc/README.md`](arc/README.md) | [`arc_agi3.tex`](arc/manuscript/arc_agi3.tex) | [outreach one-pager](arc/manuscript/gkm_one_page_summary.md), promoted artifacts |

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
