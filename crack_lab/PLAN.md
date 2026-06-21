# crack_lab: autonomous ARC-AGI-3 cracking via legs + cones

Constraint (matches the real eval): **no internet at crack time; GPUs allowed.**
So the cracking loop uses NO Claude API. It may use a *local* model.

## Architecture (the construction as the engine)

1. **Start empty.** Level 1 is simple enough to crack by **primitive best-first
   search** (discovery), ordered by hard-coded **perceptual priors** (`priors.py`)
   — not the reward (which is sparse). The prior is a "mental leg": e.g. movable
   boxes should approach the container (`box_prior`). This found wa30 L1 in ~7k
   nodes (46 primitives).
2. **Build legs.** Mine reusable action motifs from the L1 solution (the
   recurring "approach-run + interact" macros). These are the first legs.
3. **Levels 2+ : cone over legs.** Primitive BFS is now too costly (the move
   budget punishes garbage — this is the λ·C complexity pressure). Search over
   {primitives + learned legs}; legs collapse many primitives per step, so the
   solution is short enough to find in budget. **Re-anchor** at each level
   boundary; **evolve** the library by mining each cleared level.
4. **Generalise.** The legs + per-level cone-search are not path-specific; on a
   held-out similar game the same loop re-discovers / re-binds them. That is the
   colimit-cone naturality claim, tested operationally.

## Perceptual priors = mental legs (`priors.py`)

Humans crack these fast because they bring preconceptions the pixels don't label.
Hard-coded, game-agnostic, cheap; consumed by the search as objects-to-bind and
progress heuristics (never as the reward):
avatar (rigid sprite I control) · movable_objects · container (ring+interior) ·
box_prior · salient_change (blink/motion) · nonstructure_change (world reacted) ·
interaction_sites. Planned: legend/target template match, progress-bar/HUD,
symbol/pattern equality. New games extend this library, not the method.

## Local-LLM driver (idea: replace Claude with an offline model)

The expensive, judgement-heavy parts of the loop — *propose which legs to mine,
which prior/goal to bind on a new game, which cone to try first* — are currently
hard-coded heuristics. A **local LLM** (GPU, no internet) can drive these:
- input: a compact symbolic scene description (objects from `priors.py`, deltas,
  available actions, levels_completed) — NOT raw pixels.
- output: a goal hypothesis + an ordering over legs/cones to search.
- the deterministic search still VERIFIES every step against the game
  (levels_completed), so the LLM only proposes; it never substitutes for the
  reward. This keeps it honest and eval-legal (offline).
This is a clean drop-in: `priors.py` already produces the symbolic state the
local model would condition on. It is the natural next step once the heuristic
driver plateaus across games.

## Status
wa30: **L1 + L2 cracked SEQUENTIALLY from scratch**, replay-validated, by the
consolidated `gkm_crack.py` -- an abstract GKM cone-over-legs driven by a
game-specific connector that is BUILT by the local LLM (anchor + manipulation verb)
plus interaction-learning (carrier/region/border/toggle/background). The grounded
mechanic is **pick_up_and_carry** (not push). See FINDINGS.md R-CARRY + R-GENERAL.
L3 open: the avatar is walled off from the container -> needs LLM scenario
comprehension + discovery of a wall-crossing mechanic (probe abstract actions ->
bind semantically), the next build.
ls20: L1 cracked (slide). g50t: maze-gated. tr87: symbol game.

## Architecture invariant (load-bearing)
The ENGINE is abstract (legs, cone, search, replay-validation; no game facts). ALL
game-specific knowledge -- which colour is avatar/carrier/region, what attached vs
resting means, the target is a static landmark, same-colour fragments aren't boxes,
a co-worker exists -- lives in the CONNECTOR and is meant to be DISCOVERED/PROPOSED
by the local LLM (propose), with interaction + levels_completed as the VERIFIER. The
local LLM can mis-bind (e.g. it called carry "push" from a terse table; legible
natural-language trials fixed it) -- so propose->verify, never propose-only.
