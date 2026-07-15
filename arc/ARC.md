# ARC-AGI-3: replay-validated program growth

This domain applies source accounting and replay validation to programs that solve
local **ARC-AGI-3** keyboard games. It asks which retained routines are reused across
levels, which source is added, and which successful behaviors remain literal plans.
It does not measure official ARC-AGI-3 interaction efficiency.

- **Self-contained manuscript:** [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex)
  (Kolmogorov/MDL, Schmidhuber's Gödel machine + PowerPlay + curiosity, the colimit-cone
  view, the method, and results). Build with `make -C manuscript`.
- **Narrative / docs chapter:** the Sphinx page *Self-Improving Agent*
  (`docs/self_improving_agent.rst`), deployed at
  <https://sashakolpakov.github.io/gkm/>.
- **Full chronological lab log:** `FINDINGS.md` (in the code dir), including the honest
  negatives.

## The method in one paragraph

The local harness exposes `step(action) -> frame` (a 64×64 colour grid), the reward
`levels_completed`, and `clone()` for lookahead. The clone operation is a simulator
oracle supplied by this research harness; it is not part of the official ARC-AGI-3
`reset()`/`step()` agent interface. A **proposer** (a local model, or
the Claude Code agent invoked headlessly with tools + a tester) is given a rich
**human-preconception** system prompt and *writes its own* `solve(env)` program:
perception, a mechanic probe, a planner, a strategy. A candidate is admitted only if it
verifiably lowers the free energy `F = R + λ·C` on the real game (`R` = −levels reached,
`C` = description length), with the simulator as ground-truth verifier and every result
**replay-validated**. To expose retained structure, the harness enforces a growing **leg
library**: each level's player only *composes* shared skills (`legs.py`), a per-level
**debrief** refactors repeats into shared legs. The historical `marginal_C` statistic is
the positive net description-size change of the library and player files. Unchanged
legs add no charge, but same-size replacement can also add no charge; source provenance
and replay are therefore necessary to interpret the scalar.

## Current Promoted Artifacts

Replay-validated leg-library states are promoted automatically into
`crack_lab/agent_solutions/`. Other game notes remain lab/WIP context until
represented by one of these promoted artifacts.

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The ongoing bounded campaign has also promoted replay-valid partial endpoints for
`ft09` through L4 (46 actions), `sp80` through L4 (79 actions), `g50t` through L4
(145 actions), `tr87` through L4 (126 actions), and `r11l` through L4 (45 actions).
These are operational campaign artifacts, not additions to the two complete
histories reported in the manuscript table above.

The final bounded attempt also promoted `tu93` through L1 (18 actions). `sc25`
ended at the API usage limit with clean L1 WIP but no promoted level.

The definitive [Competition-Mode scorecard](https://arcprize.org/scorecards/9e166671-0953-42f3-89de-a0fd57d7b147)
scores **17.136507936507936%** across all 25 public games: 37/183 levels in 1456
API actions including resets. It replays every promoted endpoint above with the same
game-agnostic architecture.

- Historical lab notes below describe earlier runs and hypotheses; treat them as WIP
  unless they have a promoted artifact.

- On `wa30` the agent found useful level tactics: freeze the target region
  at level start; complement an autonomous helper by taking the *farthest* objects; and
  the *asymmetric carry collision* (a carried object can enter a wall cell the avatar
  cannot) that makes the L3 relay geometrically possible. **Honest audit:** the priors
  of those runs were not fully neutral — distilled from earlier human play, they named
  the carry mechanic and hinted relay-at-a-boundary. The priors have since been
  **neutralized** (generic world-priors only; no mechanic recipes, no verb names).
  A later neutral-prior run independently reached L1, but the complete L1-L9 lineage
  remains prior-contaminated and must not be described as mechanic-blind.
- The **same game-agnostic agent** transferred to `ls20` (a different mechanic) with no
  code change. Notably, ls20 got **no mechanic-name leak** (its interaction probe emitted
  only `move`), and the shared priors were wa30-flavored — *wrong* for ls20's
  transform-tile mechanic — yet the agent discovered the real mechanic itself (a generic
  clone-BFS over game state). This supports cross-game reuse of the harness, subject to
  the stronger clone-enabled interface; it is not an official sample-efficiency result.
- Under the enforced leg library, the promoted `ls20` net-growth trace is a
  **sawtooth**, not monotone: `43 → 2` and `45 → 3` are leg-reuse drops, while `45`,
  `72`, `130` are novelty spikes at mechanic transitions (drifting-HUD noise mask,
  recovered plan artifacts, the combination-lock/display family); L7 drops back to
  `67` by reusing part of the L6 lock/display understanding while adding fog-of-war
  mapping. The troughs are consistent with reuse because the corresponding source calls
  unchanged search legs and replay succeeds. The numbers alone do not detect novelty.
- The complete promoted `wa30` record is
  `112, 78, 95, 47, 405, 225, 145, 204, 147` at L1--L9. The manuscript audit sidecar
  maps the early clean Git promotions, the later preserved promotion states, and the
  final nine-level artifact to this ledger. The root checkpoint remains the unchanged
  operational resume state, so its own record list is shorter than the published history.

## Honest limitations

- The manuscript endpoints are `wa30` 9/9 and `ls20` 7/7; the repository also
  contains the partial campaign endpoints listed above. The closed official scorecard
  measures their aggregate replay coverage, but is not evidence about discovery cost
  or generalization across the full ARC-AGI-3 distribution. Recovered verified paths
  in the artifacts are charged as literals and are not compact
  mechanistic legs until a debrief refactors them (as happened for `wa30` L9).
- The 596- and 393-action values are final replay paths. Cloned exploratory steps,
  proposer calls, and compute were not counted or normalized.
- The net-growth statistic can undercount replacement because additions and deletions
  within a file cancel before the positive part is taken. A future gross-diff ledger
  would require recomputing transitions from paired snapshots and must not be mixed
  with the historical values reported here.
- The loop currently needs a **strong** proposer: a prompt-only local model mis-reasoned
  two-sided reachability under barriers even with the priors spelled out. The open
  question is how weak a proposer the same harness (priors, simulator-as-verifier,
  free-energy admission) can lift to competence.
- Sonnet repeatedly violated the declared interface during stalled `ft09` work. The
  run emitted two separate attempts at `env._game` access and private-object
  enumeration. These are cheating
  attempts in the operational sense, not merely poor solutions: they sought evidence
  unavailable through frames and actions. The exact first transcript remains at
  `agent_solutions/ft09_legs/wip_context/level_01/interrupted_a9a30e6e4da1/`, and the
  run is WIP-only. The repetition suggests compliance weakens when observational
  progress stalls. Pre-execution blocking plus an independent promotion-time taint
  check is therefore mandatory for every proposer, including Opus.

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
