# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 6 at 196 actions.
- Remaining real-action budget under the harness cap: 404 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- Using the lowest controls preserves four switches at the final shaft—enough for a down/up crossing. I’m probing the three possible column-5 catches and an uncaught descent to find which one exposes a gap in the column-6 wall.
- The lowest catch is correct: expand row 9, flip down, and step right; the avatar crosses the wall alive into column 6. I’m verifying the final two-action ascent from column 6 into column 7 now.
- Column 6 was only the midpoint. Staying in downward gravity and stepping right again lands safely in column 7 below the wall; the hazard is in column 6, not that shaft. I’m testing the upward flip from this protected column-7 landing.
- The column-7 ascent hits the upper hazard. The safe route is visible in the lower landing: walk left across its solid floor to column 5, ascend beside the hazard, then cross above it into columns 6–7.
- The remaining blocker is the right-shaft hazard: it must be approached from above, or made safe. Before extending the maze again, I’m rechecking the interaction action on this specific hazard while the avatar is in its adjacent shaft; earlier inert hazards do not prove this final object is inert.
- The final ascent was intentionally under-staged: I opened the column-5 blocker but left every earlier column-5 catch ghosted, so the avatar overshot to the sealed shaft. I’m varying those persistent catches before the column-3 descent to land at the wall gap instead.
- A remote catch now gives the desired geometry. Staging column-5 row 3 or 4 leaves columns 6–7 open at the avatar’s height; I’m testing those two landings directly instead of descending to the sealed shaft.
- The isolated color-15 object behaves exactly like the other lethal hazards, so the assumed target is wrong. I’m reproducing the validated level-6 reward transition through the same public action surface to identify the actual completion event before choosing the next level-7 subgoal.
- The verified reward transition shows color 7 is the goal; color 15 is only a hazard. The level-7 goal is still hidden, so the right shaft was a false target. I’m continuing the alternating-room climb from the safe side room and mapping which column gives the next surviving gravity transition.
- The winning transition is verified: from the staged side room, move left to column 1 and flip gravity; reward increases from 0 to 1. I’m now encoding the 60-action route as one reusable gravity-zigzag leg with dynamic lowest-button selection, then I’ll run the full harness.
- The production solver now passes: `RESULT levels=7 moves=256 replay_ok=True err=None`. I’m removing the exploratory probe script and doing one clean replay so only the reusable leg and level dispatcher remain.
- Implemented level 7 successfully. - Added `cross_staged_gravity_zigzag` in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_bp35_bp35_l7_max180_protectedlog_20260728/legs.py:624). - Added level dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_bp35_bp35_l7_max180_protectedlog_20260728/players.py:36). - Syntax checks pass. - Final harness result: `RESULT levels=7 moves=256 replay_ok=True err=None`.

## Preserved local probes

- `probe_level7.py` (5105 bytes)
- `probe_level7_alt.py` (2444 bytes)
- `probe_level7_alt_support.py` (2389 bytes)
- `probe_level7_altfinal.py` (1867 bytes)
- `probe_level7_bestfirst.py` (4108 bytes)
- `probe_level7_bfs.py` (3806 bytes)
- `probe_level7_clean.py` (1976 bytes)
- `probe_level7_collapse_trace.py` (3370 bytes)
- `probe_level7_continue.py` (11576 bytes)
- `probe_level7_controls.py` (1908 bytes)
- `probe_level7_cycle.py` (8142 bytes)
- `probe_level7_endgame_options.py` (8835 bytes)
- `probe_level7_first_supports.py` (3677 bytes)
- `probe_level7_fresh_search.py` (3766 bytes)
- `probe_level7_hazard.py` (2588 bytes)
- `probe_level7_high_left.py` (3536 bytes)
- `probe_level7_initial_combos.py` (2584 bytes)
- `probe_level7_live_search.py` (4858 bytes)
- `probe_level7_macro.py` (2995 bytes)
- `probe_level7_macro_live.py` (8406 bytes)
- `probe_level7_macro_search2.py` (5577 bytes)
- `probe_level7_macros.py` (3165 bytes)
- `probe_level7_no_control.py` (5991 bytes)
- `probe_level7_observe.py` (2524 bytes)
- `probe_level7_occluded.py` (2100 bytes)
- `probe_level7_opening_pairs.py` (2525 bytes)
- `probe_level7_persistent_beam.py` (7972 bytes)
- `probe_level7_progress.py` (5879 bytes)
- `probe_level7_room2.py` (1455 bytes)
- `probe_level7_search.py` (1024 bytes)
- `probe_level7_stage_boundaries.py` (3519 bytes)
- `probe_level7_stage_extra.py` (4625 bytes)
- `probe_level7_stages.py` (5000 bytes)
- `probe_level7_start_all.py` (2156 bytes)
- `probe_level7_switches.py` (2368 bytes)
- `probe_level7_tail.py` (5609 bytes)
- `probe_level7_tail_search.py` (3505 bytes)
- `probe_level7_transitions.py` (3150 bytes)
- `probe_level7_two_macros.py` (6007 bytes)

Run or inspect the smallest relevant probe before writing another one.
