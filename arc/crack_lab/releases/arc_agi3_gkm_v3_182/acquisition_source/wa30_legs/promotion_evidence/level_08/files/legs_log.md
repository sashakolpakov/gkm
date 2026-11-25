# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1: carry pieces onto marked cells
- Reproduced by `probe.py`: actions 1/2/3/4 move the color-14 avatar up/down/left/right by four pixels; 5 alone has no visible effect. Other pieces remain stationary.
- `probe_mechanic.py`: from entry, `[1,1,5]` picks up the piece above the avatar. Each directional move translates both cells while preserving their relative offset; 5 releases it. Approaching highlights a piece with color 3.
- `probe_goal.py` and `probe_win.py`: filling one or two of the three red cells gives no reward. Releasing the third piece onto the final cell increments levels_completed to 1. A held piece hides the red marker too, so release is required.
- `candidate_three_objects.json` preserves the independently clone-verified 27-action discovery path.
- `probe_leg.py` reproduced the first planner failure from pristine entry: it selected an already delivered piece, uncovering its original marker. The leg now preserves initial marker locations and excludes delivered pieces from pickup candidates.
- Final harness: `RESULT levels=1 moves=28 replay_ok=True err=None`.
- `fill_marked_cells` plans paired-cell translations and clone-verifies that every delivery decreases the remaining marker count before committing.

## Level 2: cooperate with an autonomous carrier
- `probe_level2.py` reproduces the color-14 avatar's four movement actions and the independent color-12 carrier advancing on all five actions.
- `probe_mechanics.py` and `probe_contexts.py` reproduce the carrier collecting, transporting (color 5), and releasing yellow pieces onto marks. Five pieces share a six-cell marked region.
- The original delivery planner fails clone verification because the carrier changes source and destination occupancy during the route.
- `probe_cooperate.py` reproduces a winning clone: retrieve the distant piece, deliver it, leave the delivery area, then yield while the carrier delivers the others. Completion increments levels_completed even with one marked cell unused.
- Reused `fill_marked_cells` with an optional cooperative mode: test bounded delivery-and-yield alternatives on clones and commit only a complete winning path. Clone probes are throttled below 300 steps/s.
- `play_level_2` only composes that existing leg.
- Final harness: `RESULT levels=2 moves=96 replay_ok=True err=None`.

## Level 3: stage pieces across a divider
- `probe_l3.py` reproduces the entry after the supervisor's 96-action prefix: avatar at (9,4), autonomous carrier at (3,12), and five pieces. Coordinates are four-pixel tile (row,column).
- All five cloned actions advance the carrier; actions 1–4 move the avatar in the four directions. Repeated rightward movement stops the avatar at column 7 against the column-8 divider.
- Waiting 31 actions leaves reward at 2 while the carrier moves the two divider pieces into the goal region at (6,13) and (8,13). The three pieces on the left remain inaccessible to it.
- Side pickup `[1,1,1,1,4,5]` from entry followed by `[4,4,4,5]` stages the piece at (5,8), with the avatar at (5,7). Moving down and yielding lets the carrier collect it and deliver it to (6,14); reward remains 2 with two pieces still on the left.
- Added `handoff_across_barrier`: infer the divider, preserve pickup offsets in routing, stage each reachable piece, leave pickup range, and verify collection. Exclude goal pieces from further manipulation and clone-verify complete reward before committing.
- Existing `fill_marked_cells` cannot express this handoff: its above-avatar cargo cannot reach across a full-height divider, and its red-mark detector includes divider cells. Earlier players remain unchanged.
- `play_level_3` only composes the new leg. `candidate_barrier_handoffs.json` preserves the validated full campaign path.
- Final harness: `RESULT levels=3 moves=187 replay_ok=True err=None`. Independent probe replay: `PROBE_RESULT 3 187 True None`.

## Level 4: rectangular handoff enclosure

Reproduced with `probe_level4.py` and `probe_mechanics.py` from the admitted
checkpoint using only public frames, actions, and clones. Avatar signature
(0,14) begins at cell (8,7); actions 1/2/3/4 move it up/down/left/right.
Action 5 picks up an adjacent faced piece: [1,3,1,5] removes the piece at
(6,6), while the avatar remains (7,6). The following left move is blocked by
the divider. Other carriers (colour 12) act during these actions; waiting
also lets the exterior piece reach an exterior mark.

The enclosure has divider rows 5 and 11, columns 5 and 10, with six interior
pieces and seven exterior marks. Existing handoff logic applies. Its prior
line detector required more than half the board width in visible cells,
so it missed this smaller enclosure. Detect segments using at least three
visible barrier cells and their observed endpoints instead. No new leg.
The complete handoff plan succeeds on a pristine level-4 clone.

Validation: resumed and fresh `gkm_try.py` both report
`RESULT levels=4 moves=280 replay_ok=True err=None`.
`verify_candidate.py` independently replays from level 1, observes exterior
mark count eventually reach zero, validates the resulting path, and writes
`candidate_wa30_four_levels.json`. Checkpoint remains supervisor-owned.

Level 5: completed with `deliver_with_moving_carrier`.
- Reproduced entry by replaying the admitted checkpoint through public actions:
  level 4 complete, 280 moves. Avatar color 14 at tile (9,11); actions 1..4
  translate up/down/left/right. Action 5 picks up and drops adjacent cargo;
  the pickup offset persists during translation.
- `probe5.py`: unattended carrier color 12 acts every turn. Waiting reaches
  terminal at turn 125 with four pieces settled. Pure color-5 walls block the
  avatar; the two-row opening permits passage subject to cargo occupancy.
- `delivery_probe.py`: static plans collide with the moving carrier and newly
  filled targets. `dynamic_probe.py`: replanning, remembered target positions,
  two blocked translations followed by drop-and-retreat permit a handoff.
  Clearing the carrier's route is necessary for further autonomous deliveries.
- Pristine-entry clone search found a reward-verified 112-action completion.
  Saved suffix: level5_dynamic_candidate.json. Saved full replay candidate:
  wa30_level5_candidate_path.json (392 actions). Checkpoint untouched.
- New leg searches bounded source orders, navigates from current observations,
  and commits only a clone plan that increases levels_completed. Level player
  consists solely of the imported leg call; earlier legs remain unchanged.
- Both `python gkm_try.py` and `GKM_FRESH_REPLAY=1 python gkm_try.py` returned:
  RESULT levels=5 moves=392 replay_ok=True err=None.

## Level 6: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_wa30_arc_agi3_n0_fresh_frontier_19jl4di6/outlined_delivery_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 7: let the carrier settle, remove it, then fill outlined targets
- Replayed the admitted checkpoint through the public API: entry has six completed levels after 439 moves. Directional clone probes moved color 14 in the corresponding four directions; all five actions advanced color 15.
- Entry target signatures are `(2,9)` at cells `(7,3)` and `(8,3)`, with plain `(2,)` cells at `(7,12)` and `(8,12)`. Loose cargo begins at `(6,8)` and `(8,7)`.
- Immediate application of `clear_interfering_carrier` fails its 60-step clone bound: the actor moves between facing and USE. Waiting from pristine entry lets it deliver cargo right and become stationary with the other cargo. After 22 USE waits, `[4,4,1,4,5]` removes it and releases its cargo; `probe_level7.py` preserves this reproduction.
- Added optional `settle_turns` to the existing removal leg: wait on its clone until carrier and carried-cargo signatures remain unchanged, then use its existing approach/removal algorithm. The entire removal must verify before committing.
- Added optional `persistent_targets` to `fill_marked_cells`: preserve initial destinations and count cargo occupancy, because the avatar can hide the second target after the first delivery. Without it, the leg returns prematurely at reward 6. Both outlined targets filled after removal advance reward to 7; plain targets are not required by the winning path.
- `play_level_7` only composes these two existing legs. Their default behavior for earlier players is unchanged.
- `verify_level7.py` independently records and validates the composition. `settled_carrier_delivery_candidate.json` contains the full 495-action path for supervisor adoption; checkpoint.json was not modified.
- Final harness: `RESULT levels=7 moves=495 replay_ok=True err=None` (56 new real moves, 105 moves below the cap).

## Level 8: intercept opposing carriers, assist two deliveries, then yield

Reproduced through the public API by `probe8.py`, `probe8_dynamics.py`,
`probe8_reuse.py`, `probe8_intercept.py`, `probe8_pairs.py`, and
`probe8_wait_finish.py`. Coordinates below are four-pixel (row, column) cells.

- The admitted campaign replay enters level 8 after 495 actions. Its avatar
  starts at (8,1); cloned actions 1/2/3/4 move up/down/left/right. Action 2
  is blocked there by the row-9 wall. All five keys advance other actors.
- Wall rows 6 and 9 have openings at columns 4..5 and 9..10 respectively.
  Two color-12 and two color-15 actors start in the upper and lower regions.
  Waiting shows color 12 carrying cargo toward outlined `(2,9)` cells on the
  right, and color 15 carrying it back toward plain `(2,)` cells on the left.
  The carrier's cargo has signature `(5,9)` while in transit.
- The old approach/face/USE removal method fails its 80-step pristine-entry
  clone test. A bounded beam guided by wall-respecting distance finds a
  seven-action interception of the upper color-15 actor, then a thirteen-action
  interception of the lower one. Each interception verifies actor removal.
- Removing those actors and yielding alone does not finish within 105 turns.
  The upper carrier needs help with the clustered left cargo and its congested
  destinations. Downward pickup preserves cargo below the avatar, allowing a
  route across the upper side of the delivered cluster.
- Explicit assistance pairs (2,2)->(2,14), (2,1)->(2,13) fill two outlined cells.
  Replanning each translation avoids carriers moving into the route. The
  existing delivery skill verifies pickup and reduced unfilled target count
  before committing each delivery. The two deliveries finish at suffix turn 86.
- Yielding with UP moves the avatar away from the cargo and then waits against
  the upper edge. Other carriers finish the remaining deliveries; reward
  increases from 7 to 8 at suffix turn 110. A held or highlighted piece alone
  is not used as the completion criterion.
- Reusing interception on a pristine level-7 clone removes its actor in 11
  moves; existing outlined-target delivery then reaches level 7 at campaign
  move 485, ten moves earlier than the admitted checkpoint. The supervisor
  checkpoint was never edited. Prefix optimization is tested from level 1.

Implementation: `clear_interfering_carrier(..., anticipate_motion=True)` adds
bounded clone interception to the existing removal skill. Optional ordered
`delivery_pairs` in `fill_marked_cells` select and verify assistance deliveries.
The small new `yield_until_complete` leg verifies reward before committing waits.
`play_level_8` only composes these imported legs.

Validation: `GKM_FRESH_REPLAY=1 python gkm_try.py` reports
`RESULT levels=8 moves=595 replay_ok=True err=None`.
`verify_level8.py` preserves a separately validated full action path as
`wa30_eight_levels_595_candidate.json` for independent harness adoption.
