# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The alternate branch retains the three-action lead through another carrier exchange: it reaches action 47 after moving the staged bridge back through the left carrier. The heuristic remains stable, so I’m advancing until it either rejoins a validated state or becomes a cycle.
- The swapped branch reaches action 56, but its local lead has evaporated; it has just staged a payload at `(54,42)`, whereas the original crosses the first wrap by action 50. Before abandoning it, I’m checking whether this state exactly matches a much later validated state—the heuristic’s remaining estimate is only 24, which would make a suffix splice decisive if exact.
- The action-56 state is not an exact trace match, but it is visually close to a late phase. I’m testing every validated macro suffix directly from this swapped-payload state, shortest total first. Reward replay—not similarity—will decide whether any suffix splice is valid.
- No validated suffix works from action 56. The branch’s next carrier transfer reaches action 67, still with a low 22-action similarity estimate but now only 45 actions of useful budget remain. I’m testing suffix splices at this new state before paying for another closure.
- At action 75 the peg unloads back at its original entry coordinate after six left keys. This looks more like a carrier cycle than progress, so I’m checking exact-state recurrence and suffix compatibility now; if it matches an earlier state at higher cost, the swapped branch is dominated and can be discarded.
- The action-75 state has no validated suffix splice, but its estimated remaining structure is still small enough that I’m advancing one more carrier cycle. The branch must win by roughly action 112 to close the campaign gap; otherwise it is no longer useful.
- The bridge-first branch is now falsified: after action 75 its entire 64-state carrier closure has no legal non-reversing payload move. I’m returning to the validated peg-first state at action 40 and using the macro planner only on the open wrapped section, where alternate peg/bridge move order—not carrier navigation—is the remaining source of possible savings.
- From action 40, the planner over-prioritized a carrier re-entry and began undoing the staged peg instead of considering the immediate leapfrog moves. I’m stopping that branch and switching this phase to “immediate legal move first”; key closure will be used only when the board has no coordinate move.
- The open-board search confirms the validated sequence is nearly forced: after the first peg advance, the only alternative is moving the bridge backward, and that branch enters carrier cycles instead of rejoining reward. I’m stopping this exploratory run; it has not produced a valid shorter level-7 leg.
- I’m cleaning only the exploratory files created in this attempt that produced no unique winning data. I’ll keep the compact reproducer, entry-model proof, per-level minimizer, bidirectional key proof, full-campaign verifier, and all three optimized candidate JSONs; the discarded findings are summarized in `frontier_brief.md`.
- The exploratory artifacts have been removed; they were all created in this attempt and contained no unique winning path. The retained probes are compact and reproducible. I’m doing the final three checks now: level-9 leg equivalence, 633-action clone campaign, and the resumed real harness cap result.
- Level 9 is implemented and clone-verified, but the real campaign remains blocked by the 600-action cap. - Added level-9 dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont2_max180_20260728/players.py:45). - The existing level-9 leg wins in exactly 102 actions and matches the preserved candidate’s final frame. - Optimized the known campaign from 646 to 633 actions; preserved it as [campaign_candidate_633.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont2_max180_20260728/campaign_candidate_633.json). A pristine clone completes level 9 at action 633. - Recorded all verified observations and falsified shortcuts in [frontier_brief.md](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont2_max180_20260728/frontier_brief.md). Final real harness result: ```text RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap ``` The supervisor checkpoint was untouched. Its 544-action prefix leaves 56 actions for a 102-action level, a 46-action gap. Even adopting the independently optimized 531-action prefix leaves a further 33 actions to remove.

## Preserved local probes

- `campaign_candidate_633.json` (11978 bytes)
- `level4_ddmin_50.json` (1085 bytes)
- `level6_ddmin_92.json` (2021 bytes)
- `level9_candidate_102.json` (1355 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `probe_ddmin_level.py` (2912 bytes)
- `probe_level9_abstract_entry.py` (3812 bytes)
- `probe_level9_reproduce.py` (4363 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (1755 bytes)

Run or inspect the smallest relevant probe before writing another one.

## Verified continuation observations

- `play_level_9` now composes the existing
  `solve_multi_bridge_wrapped_carrier_peg_solitaire` leg.  A pristine
  level-9 clone still completes at exactly 102 actions.
- The level-9 entry abstraction has 7 reachable goal layouts through depth
  18: one at 14 lattice moves, three at 15, two at 16, and one at 18.  The
  unique 14-move layout is shortest.  Replaying the known 74-action relay
  suffix from every layout wins, but none wins earlier.
- Selection is cleared by a real carrier move.  Action 7 observationally
  undoes the immediately preceding action; it did not provide a transport
  shortcut.
- Wrapped-screen long-jump probes found only ordinary 12-pixel lattice
  jumps.  No cross-edge click destination was legal.
- At known action 62, two left keys visually align the mobile peg and a
  bridge with the stationary peg, but `(30,28)` is not yet a landing slot;
  the apparent 70-action shortcut is invalid.
- The coordinate-only finale at known action 84 has 32 reachable visible
  piece states through depth 8 and no reward.  The verified 9-lattice-move
  finale is locally minimal under the observed move surface.
- Unloading the anchored peg at known action 28 exposes an empty carrier
  that moves only horizontally for eight right keys.  No vertical carrier
  move appears, and early lower-bridge staging does not persist into the
  remote vertical lane.
- Integrated the reward-verified 50-action direct level-4 leg and the
  one-key level-6 deletion into `legs.py`; the level-4 action tuple exactly
  matches `level4_ddmin_50.json`.
- Final resumed harness result after dispatch:
  `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError:
  real-move/time cap`.  The fixed 544-action checkpoint still leaves only
  56 actions for level 9.  `checkpoint.json` was not modified.
