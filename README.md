# Gödel–Kolmogorov Machine: Auditable Program-Growth Experiments

The Gödel–Kolmogorov Machine is a collection of verifier-driven program-growth
experiments. In the ARC-AGI-3 study, the Gödel–Kolmogorov Machine uses a proposer
to write solver code and a simulator to validate promoted behavior by fresh replay.
A source-size ledger records positive net growth of the retained library and player
files.

GKM is universal at the producer level: every public game uses the same fixed
proposer contract, Arena interface, blank scaffold, complexity coordinate, and replay
gate. Its learned executable outputs are necessarily game- and level-dependent; that
specialization is what the universal producer generates and promotes.

The result is not only a solved-level count, but an audit trail of how competence
was acquired: promoted solver code, replay validation, WIP snapshots, charged
literals, reusable solver-leg refactors, and marginal-complexity accounting.
The name Gödel–Kolmogorov Machine joins verifier-gated self-revision with
description-length selection; it does not claim proof-search optimality or exact
Kolmogorov complexity. After this full introduction, the repository uses **GKM**
as its abbreviation.

## Current replay-validated ARC-AGI-3 artifacts

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Scope | Verified levels | Stored replay actions | Ledger charge |
|---|---:|---:|---:|
| Frozen v2 release (25 games) | 181/183 | 7001 | — |
| `wa30` uniform history | 9/9 | 597 | 318 |
| `ls20` uniform history | 7/7 | 365 | 760 |

The frozen release contains one checkpoint record for every promoted level. The generated all-game marginal table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The frozen v2 [Competition-Mode scorecard](https://arcprize.org/scorecards/cf75e14b-2c25-41cb-bc70-53bd57411edb)
scores **98.11664037825032%** across all 25 public games. Its distinct unweighted
coverage is **181/183 = 98.907103825137%**. The certified paths contain 7001 actions;
the scorecard used 7069 API actions including resets. The corresponding
[ONLINE shakedown](https://arcprize.org/scorecards/e293eeae-c0de-4263-a916-0a40ad282cbc)
preceded the definitive Competition replay. The single source of truth for the
campaign, its milestones, and release gates is
[`arc/crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md`](arc/crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md).
Machine-readable promoted artifacts and audit outputs live under
[`arc/crack_lab/agent_solutions/`](arc/crack_lab/agent_solutions/) and
[`arc/crack_lab/audit_results/`](arc/crack_lab/audit_results/).
The automatic policy uses clean retries at one unchanged frontier as its
game-independent complexity coordinate for both effort escalation and the
later independent observation-only sidecar role. At hard frontiers, a separate
supervisory proposer may turn authenticated native/sidecar evidence into a
quarantined tactical handoff, but cannot select work or promote a solver; see
[`REPRODUCE_ARC.md`](REPRODUCE_ARC.md) for the machine-readable policy check.
The scheduler may allocate a sidecar slot but cannot write its tactical brief:
every contiguous sidecar must be bound to an authenticated same-frontier
native-proposer request or an admitted supervisory handoff.

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

The definitive all-game card used the complete frozen release:

```bash
python arc/crack_lab/replay_scorecard.py --mode competition
```

Canonical endpoint folders are under
[`arc/crack_lab/agent_solutions/`](arc/crack_lab/agent_solutions/). The complete
publication histories for `wa30` and `ls20` are additionally indexed under
[`arc/manuscript/artifact_history/`](arc/manuscript/artifact_history/).

## Exact claim

The two complete published histories are:

- `wa30`: 9/9 levels, a 597-action validated replay, and the uniform clean
  reacquisition ledger `43, 20, 32, 50, 39, 23, 28, 34, 49`, totaling 318.
- `ls20`: 7/7 levels, a 365-action validated replay, and the uniform clean
  reacquisition ledger `40, 54, 86, 114, 138, 170, 158`, totaling 760.

Every entry is tied to its own replay-validated promotion manifest and exact
winning-source boundary. Earlier nonuniform histories remain superseded provenance,
not inputs to these canonical ledgers.

Do not infer a live campaign total from this historical pair. Current solved-boundary
statistics are regenerated in `arc/crack_lab/audit_results/`; the campaign plan
defines the uniform budget, escalation, taint, replay, and promotion policy.

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
| [`arc/README.md`](arc/README.md) | [`arc_agi3.tex`](arc/manuscript/arc_agi3.tex) | [manuscript/reproduction bundle](arc/manuscript/README.md), [outreach one-pager](arc/manuscript/gkm_one_page_summary.md), promoted artifacts |

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
