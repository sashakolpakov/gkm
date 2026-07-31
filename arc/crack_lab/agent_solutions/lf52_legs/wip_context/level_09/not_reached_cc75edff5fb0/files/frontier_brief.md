# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The hybrid search has the efficient first-region behavior back: 96 states at cost 13. It will preserve turn distinctions only once a capture creates offscreen divergence.
- The first capture frontier is imminent at cost 15/182 states. After it, the turn-sensitive key should retain both the 22- and 23-action visually identical states.
- The turn-sensitive post-capture frontier is at cost 20 with 248 states. Both timing variants are now retained; I’m continuing through the wrapped reveal.
- The singleton alignment reached cost 21 with 292 states. The next two actions should distinguish the false early reveal from the verified timing.
- Both 22- and 23-action reveal timings are retained; the search is at cost 26 with 335 states. I’m continuing toward the four-peg region, which will identify the viable branch.
- The two-peg transit is at cost 27 with 356 states. No cap or error; the viable timing is still in the queue.
- Cost 28 has 392 turn-sensitive states. The search remains bounded and is preserving the verified timing branch.
- The cost-28 layer is broad and wall time is becoming the limiting factor. I’ll take the next completed layer; if the four-peg transition still has not appeared, this search formulation is not viable within the harness cap.
- I’m testing one remaining direct geometry at the loaded carrier: after seven right moves, the remote peg’s column aligns with its apparent row-18 bridge. If a downward jump is legal there, it bypasses the edge relay entirely.
- The seven-right direct alignment is falsified: only the two pegs are selectable and no downward jump exists. I’m consolidating now—verifying the compact level-4 leg exactly matches its reward-verified JSON, compiling the solver, and removing exploratory probes that produced no winning path.
- The final resumed verification is honest and unchanged at the hard boundary: `levels=8`, `moves=600`, replay-valid. I’m preserving the newly verified falsifications and code changes in the frontier brief so the next clean continuation can avoid repeating these probes; the supervisor checkpoint remains untouched.
- I could not reach `levels>=9` within the hard move cap. Implemented: - Level-9 dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont3_max180_20260728_ix3e221x/players.py). - Verified 50-action level-4 leg and one-action level-6 optimization in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont3_max180_20260728_ix3e221x/legs.py). - Preserved verified observations in [frontier_brief.md](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont3_max180_20260728_ix3e221x/frontier_brief.md). Final harness result: ```text RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap ``` The fixed 544-action checkpoint leaves 56 actions, while the verified level-9 leg requires 102. `checkpoint.json` was untouched.

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
- `wip_glue_notes.md` (3093 bytes)

Run or inspect the smallest relevant probe before writing another one.

## Clean continuation verification

- Reproduced the pristine level-9 entry and the existing multi-bridge leg:
  `level9_candidate_102.json` earns `levels_completed=9` on a clone in exactly
  102 actions.
- Reproduced the prefix reductions independently: the direct level-4 leg wins
  in 50 actions and the current level-6 leg matches the 92-action candidate.
- Composed those legs in `players.py` and added `play_level_9`, which only
  calls `solve_multi_bridge_wrapped_carrier_peg_solitaire`.
- Default checkpoint verification after composition is still the hard boundary:
  `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap`.
- Verified high-value falsifications from pristine states: action 7 is a
  recorded one-input undo; coordinate action 6 cannot relocate an empty
  carrier; earlier grid/wrapped legs do not solve levels 7 or 9; moving the
  loaded level-9 peg directly from `(36,22)` to `(36,10)` is legal but strands
  the remote peg behind an unreachable two-bridge landing.
- The verified reduced campaign is still 633 actions (531 through level 8 plus
  the 102-action level-9 leg). `checkpoint.json` was not changed.
