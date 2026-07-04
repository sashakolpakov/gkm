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

As of the 2026-07-04 cleanup, replay-validated leg-library states are promoted
automatically into `crack_lab/agent_solutions/`. Other game notes remain lab/WIP
context until represented by one of these promoted artifacts.

| game | status | artifact |
|---|---|---|
| `ls20` | L1-L4 replay-validated | `crack_lab/agent_solutions/ls20_legs/` |
| `wa30` | L1-L3 replay-validated; WIP for L4+ | `crack_lab/agent_solutions/wa30_legs/` |
| `sp80` | WIP / separate concurrent run | not currently promoted |

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
- Under the enforced leg library, per-level **marginal novelty collapses** as legs are
  reused — on the current promoted `ls20` artifact: `70 → 2 → 2 → 0`. Later levels are
  near-pure composition.
- The same enforced library on `wa30` (L1–L3 validated) shows the honest complement:
  marginal novelty does **not** collapse (`112 → 78 → 95`) because each `wa30` level
  introduces a genuinely new mechanic (autonomous helper at L2; dividing wall +
  asymmetric-carry relay at L3), and the agent captured each as new named legs
  (`yield_to_helper`, `wall_col`, `relay_to_helper`, …). Reuse-collapse is a property
  of the *game's* level structure; the method pays for novelty exactly when the game
  demands it — which is what `F = R + λ·C_marginal` is for.

## Honest limitations

- The current promoted repo artifacts are `ls20` L1-L4 and `wa30` L1-L3. Higher
  `wa30` levels and `sp80` remain WIP unless represented by promoted artifacts.
- The loop currently needs a **strong** proposer: a prompt-only local model mis-reasoned
  two-sided reachability under barriers even with the priors spelled out. The open
  question is how weak a proposer the same harness (priors, simulator-as-verifier,
  free-energy admission) can lift to competence.

## Code

The cracking code lives in [`crack_lab/`](crack_lab/) within this domain.
Key modules: `gkm_arena.py` (the rawest substrate + free-energy admission),
`gkm_solve_agent.py` (proposer = Claude with discovered context + tools + tester),
`gkm_legs.py` (enforced leg-library orchestration + marginal-C accounting),
`gkm_crack.py` (the earlier discovered-connector cone), `gkm_discovery.py` (interaction
probe). Agent-written solvers are archived under `agent_solutions/`.
