# Unverified frontier brief: ls20 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 6 at 298 actions.
- Remaining real-action budget under the harness cap: 302 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The target is now directly identified: the bottom-right 3×3 hollow colour-11 ring, matching the object entered on level 6’s rewarding final move. The prior graph falsely stopped one row early because my avatar extractor classified rows 50–54 as HUD; I’m correcting that observational key and testing the two final chamber approaches from pristine entry before implementing.
- Entering the bottom-right ring with the initial state refills to 21 but does not reward, whereas the identical contact completes level 6 after its preparation route. That verifies a state-gated target. The shape and colour dials each have two reachable approach axes, so I’m measuring both transition directions; this should expose the exact finite state graph needed to prepare the target state.
- The non-rewarding contact is a one-use refill, and the level contains six such hollow rings. This makes the dense objective concrete: decrease the remaining-ring count to zero. I’m extracting the verified movement/portal graph once, then solving the six-ring ordering with the real 21-step refill constraint and replay-checking the shortest candidate on a clone.
- The six-ring tour does not reward, so rings are confirmed as energy only. The preserved mismatch idea remains live: the goal tile is the hollow colour-11 glyph, while our two reachable dials alone generate only 24 nonmatching states. I’m now isolating the other small coloured/black objects under single successful moves; inverse moves had returned them to their original pixels, which is consistent with synchronously controlled agents and would explain the otherwise unreachable target state.
- The right corridor contains a second verified shape transformer: crossing upward from row 27 to 22 changes the HUD from the T glyph to a left-facing T, while adjacent moves do nothing. I’m cycling this transformer next; unlike the lower six-state orbit, it may supply the hollow target shape. I’ll then scan the analogous remote colour transformer for colour 11.
- None of the lower/remote glyph combinations reward, so I’m checking a more physical alternative against the verified level-6 transition: exact meter timing. Its long route contains deliberate-looking oscillations, and the frontier geometry repeatedly highlighted 21-step refill budgets. I’ll record the meter immediately before reward and each refill interval; if the final ring must be entered on a boundary step, that gives a direct dense timing constraint for level 7.
- Level 6 reaches the same ring with five meter units left, so timing is falsified. The decisive remaining context is entry direction/state of the ring itself: level 6 enters it from above, while level 7’s obvious route enters from the left. I’m testing all post-contact actions and a leave/re-enter sequence to see whether the first contact opens a directional passage or changes the chamber.
- The remote transformer is phase-dependent: the same prepared shape produces different outputs when reached four successful moves later. One observed intermediate exactly matches level 6’s winning mask, but in colour 8. I’m reproducing that phase with colour 9 by adding a four-move ordinary oscillation, then leaving the transformer immediately and returning to the goal without crossing another modifier.
- The reproduced level-6 HUD state still does not complete level 7, so glyph equality is not the terminal condition. I’m broadening the reward comparison across the already verified early levels: capturing the actual contacted object and entry direction on each reward will distinguish a universal target affordance from a level-specific checkpoint/history requirement.
- The reward contacts across levels are patterned 3×3 tiles, not generic rings. I’m now comparing each pre-contact HUD glyph to the exact destination tile computed from the verified movement delta. This will settle whether completion requires equality, a directional transform, or merely contact—and will give level 7 an exact target state if equality holds.
- The cross-level comparison is now exact: every rewarding move enters a 3×3 tile identical to the current HUD (zero pixel mismatch). Level 7 therefore requires `BBB/B3B/BBB`. Since the obvious lower palette cannot produce colour 11, I’m cataloguing every aligned 3×3 special pattern exposed across the 62-position map; this should locate the remote palette or transformer instead of searching arbitrary histories.
- The pattern catalog resolves the puzzle: the hollow colour-11 tiles are refills; the actual goal is the isolated colour-8 glyph at row 51, column 30. A reproduced remote sequence yields that exact HUD (`8.8/88./.88`) with 12 energy remaining. The portal graph has a 12-move route from there to enter the matching goal from above, so I’m verifying this exact-budget chain now.

## Preserved local probes

- `probe_l7.py` (29034 bytes)
- `probe_level.py` (1532 bytes)
- `probe_reward.py` (5702 bytes)

Run or inspect the smallest relevant probe before writing another one.
