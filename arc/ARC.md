# ARC-AGI-3: a self-improving agent under free energy

This domain applies the GKM free-energy view to live **ARC-AGI-3** keyboard games,
played locally and offline through their runtime. The question: can an agent *figure a
game out on its own* — discover its perception, mechanics, goal, and a winning strategy
— from the rawest interface, carrying only general human preconceptions, with a single
free-energy rule deciding what structure is kept?

- **Self-contained manuscript:** [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex)
  (Kolmogorov/MDL, Schmidhuber's Gödel machine + PowerPlay + curiosity, the colimit-cone
  view, the method, and results). Build with `make -C manuscript`.
- **Narrative / docs chapter:** the Sphinx page *Self-Improving Agent*
  (`docs/self_improving_agent.rst`), deployed at
  <https://sashakolpakov.github.io/gkm/>.
- **Full chronological lab log:** `FINDINGS.md` (in the code dir), including the honest
  negatives.

## The method in one paragraph

The engine exposes only `step(action) -> frame` (a 64×64 colour grid), the reward
`levels_completed`, and `clone()` for lookahead — nothing game-specific, because that is
the only boundary that transfers across game *types*. A **proposer** (a local model, or
the Claude Code agent invoked headlessly with tools + a tester) is given a rich
**human-preconception** system prompt and *writes its own* `solve(env)` program:
perception, a mechanic probe, a planner, a strategy. A candidate is admitted only if it
verifiably lowers the free energy `F = R + λ·C` on the real game (`R` = −levels reached,
`C` = description length), with the simulator as ground-truth verifier and every result
**replay-validated**. To make later levels cheap, the harness enforces a growing **leg
library**: each level's player only *composes* shared skills (`legs.py`), a per-level
**debrief** refactors repeats into shared legs, and `C` is scored **marginally** (new
legs only — a reused leg is free), so parsimony rewards transfer. This is the
colimit-cone made operational: legs written by the proposer, composed by a cone, priced
by the same free energy.

## Current Promoted Artifacts

Replay-validated leg-library states are promoted automatically into
`crack_lab/agent_solutions/`. Other game notes remain lab/WIP context until
represented by one of these promoted artifacts.

| game | status | actions | total marginal C | artifact |
|---|---|---|---|---|
| `wa30` | **9/9** replay-validated | 596 | 1243 | `crack_lab/agent_solutions/wa30_legs/` |
| `ls20` | **7/7** replay-validated | 393 | 362 | `crack_lab/agent_solutions/ls20_legs/` |
| `sp80` | WIP / separate concurrent run | — | — | not currently promoted |

- Historical lab notes below describe earlier runs and hypotheses; treat them as WIP
  unless they have a promoted artifact.

- On `wa30` the agent found level tactics beyond its priors: freeze the target region
  at level start; complement an autonomous helper by taking the *farthest* objects; and
  the *asymmetric carry collision* (a carried object can enter a wall cell the avatar
  cannot) that makes the L3 relay geometrically possible. **Honest audit:** the priors
  of those runs were not fully neutral — distilled from earlier human play, they named
  the carry mechanic and hinted relay-at-a-boundary. The priors have since been
  **neutralized** (generic world-priors only; no mechanic recipes, no verb names);
  re-cracking `wa30` from scratch under neutral priors is the discriminating experiment.
- The **same game-agnostic agent** transferred to `ls20` (a different mechanic) with no
  code change. Notably, ls20 got **no mechanic-name leak** (its interaction probe emitted
  only `move`), and the shared priors were wa30-flavored — *wrong* for ls20's
  transform-tile mechanic — yet the agent discovered the real mechanic itself (a generic
  clone-BFS over game state). So ls20 succeeded **despite** misleading priors: robustness,
  not leakage. The `sp80` liquid-pour result (below) is the same story on a third game.
- Under the enforced leg library, the promoted `ls20` marginal-complexity trace is a
  **sawtooth**, not monotone: `43 → 2` and `45 → 3` are leg-reuse drops, while `45`,
  `72`, `130` are novelty spikes at mechanic transitions (drifting-HUD noise mask,
  recovered plan artifacts, the combination-lock/display family); L7 drops back to
  `67` by reusing part of the L6 lock/display understanding while adding fog-of-war
  mapping. Marginal free energy acts as a **novelty detector**.
- The same enforced library on `wa30` (now 9/9) shows the honest complement: marginal
  novelty does **not** collapse (`30, 87, 405, 225, 145, 204, 147` for the levels the
  checkpoint audits) because each `wa30` level keeps introducing new logistics
  structure — ferry/yield composition at L4, live-frame courier pacing at L5, a shared
  higher-order `neutralize_then_deliver` pattern across L6–L8, and at L9 a recovered
  61-action suffix debriefed into the reusable `grab_carry_release` / `ferry_each`
  legs. Reuse-collapse is a property of the *game's* level structure; the method pays
  for novelty exactly when the game demands it — which is what `F = R + λ·C_marginal`
  is for.

## Honest limitations

- The current promoted repo artifacts are `wa30` 9/9 and `ls20` 7/7 on the local
  preview games — not the full ARC-AGI-3 distribution and not a private benchmark
  score. `sp80` remains WIP unless represented by a promoted artifact. Recovered
  verified paths in the artifacts are charged as literals and are not compact
  mechanistic legs until a debrief refactors them (as happened for `wa30` L9).
- The loop currently needs a **strong** proposer: a prompt-only local model mis-reasoned
  two-sided reachability under barriers even with the priors spelled out. The open
  question is how weak a proposer the same harness (priors, simulator-as-verifier,
  free-energy admission) can lift to competence.

## Index of this domain

**Documents**

- [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex) — the self-contained
  manuscript (build with `make -C manuscript`).
- [`manuscript/gkm_one_page_summary.md`](manuscript/gkm_one_page_summary.md) /
  [`.tex`](manuscript/gkm_one_page_summary.tex) — the outreach one-pager
  (claim, artifacts, baseline comparison, collaboration request).
- Reports: [`arc_local_gkm_report.md`](arc_local_gkm_report.md),
  [`arc_scene_atom_discovery_report.md`](arc_scene_atom_discovery_report.md),
  [`arc_goal_induction_report.md`](arc_goal_induction_report.md),
  [`arc_leg_discovery_report.md`](arc_leg_discovery_report.md),
  [`arc_live_report.md`](arc_live_report.md).
- Full chronological lab log with the honest negatives:
  [`crack_lab/FINDINGS.md`](crack_lab/FINDINGS.md).

**Library modules** (in this directory)

- [`arc_agi3_adapter.py`](arc_agi3_adapter.py) — the offline (`LocalArcEnv`) and
  live (`ArcEnv`, scorecard-capable) connectors behind the rawest interface.
- [`arc_scene_atoms.py`](arc_scene_atoms.py) — predicate@colour scene atoms.
- [`arc_goal_induction.py`](arc_goal_induction.py) — goal induction lifted onto ARC.
- [`cone_leg_discovery.py`](cone_leg_discovery.py) /
  [`cone_leg_composition.py`](cone_leg_composition.py) — cone-leg discovery and
  composition over the ARC connector (built on [`../cone/`](../cone/CONE.md)).

**Runnable experiments** — `run_arc_local_gkm.py`, `run_arc_live_probe.py`,
`run_arc_goal_induction.py`, `run_arc_leg_discovery.py` (each `python3 arc/run_...`).

**Tests** — `test_arc_*.py`, `test_cone_leg_discovery.py`, plus the crack-lab unit
tests `test_object_mdl.py`, `test_powerplay.py`, `test_universal_crack_boundary.py`;
run `python -m pytest arc/ -q` from the repo root (heavier crack-lab loop tests live
in `crack_lab/test_gkm_*.py`).

**The cracking lab** — [`crack_lab/`](crack_lab/): `gkm_arena.py` (the rawest
substrate + free-energy admission), `gkm_solve_agent.py` (proposer = Claude with
discovered context + tools + tester), `gkm_api_agent.py` (Messages-API proposer),
`gkm_legs.py` (enforced leg-library orchestration + marginal-C accounting +
interruption-proof WIP recovery), `gkm_crack.py` (the earlier discovered-connector
cone), `gkm_discovery.py` (interaction probe). Promoted replay-validated artifacts
live under [`crack_lab/agent_solutions/`](crack_lab/agent_solutions/)
(`wa30_legs/`, `ls20_legs/` — each with `checkpoint.json`, `players.py`, `legs.py`,
and preserved WIP snapshots).
