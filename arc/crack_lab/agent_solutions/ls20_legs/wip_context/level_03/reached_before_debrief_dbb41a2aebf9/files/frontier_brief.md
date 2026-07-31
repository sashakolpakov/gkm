# Unverified frontier brief: ls20 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 2 at 58 actions.
- Remaining real-action budget under the harness cap: 542 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The solved-level geometry clarifies the failure: level 2’s target has one floor row between it and the avatar, while level 3’s target is behind a colour-5 barrier at row 50. The portal places the avatar directly above that barrier, so `DOWN` is a door interaction, not ordinary contact. I’m testing repeated pressure across the remaining energy phases to see whether the barrier is timed or requires sustained interaction.
- The barrier is neither timed nor breakable by repeated input. The central 3×3 object is asymmetric (its colour-14 pixels point right), and all prior contacts approached it from above. I’m now testing it as a directional portal/switch by entering from the left, right, and below; this is the strongest remaining context-dependent affordance and directly addresses the isolated target chamber.
- The directional context is decisive: entering the central object from above/left/right changes the HUD avatar to colour 9, while entering from below leaves it in the original colour-12 state. The below approach costs seven moves, and the exact route from there through the portal to the target costs fourteen—precisely the 21-move refill budget. This is the intended cooperation/polarity constraint; I’m verifying the complete route.
- The below-state hypothesis is negative, but the probe still shows the central contact encodes direction in the HUD/state. I’ve only carried the “above” and “below” variants through to the door; I’m now completing the left-entry and right-entry variants too. Both fit the budget exactly, and the object’s right-pointing colour-14 pattern makes left-to-right traversal especially plausible.
- The HUD colour change tracks the last movement direction, so it was not a switch state. I’m inspecting the target chamber’s exact 15×15 raw boundary now; the coarse five-pixel map hid whether there is a side opening or a portal edge. This should determine the reachable entrance geometrically instead of by further state guessing.
- The solved level-2 route reveals the ordering rule: it touches the black checkpoint, then takes the final refill, then reaches the target. Level 3’s sealed chamber therefore likely requires touching the central checkpoint before—not after—the final refill. A feasible route is upper refill → central checkpoint → lower refill → portal → target, with 13 moves after the last refill. I’m testing that ordering now.
- The checkpoint-before-refill ordering may require using the final “empty” step to land on the refill. The meter has 21 visible moves, but we only tested an empty step against a wall; landing on a ring on move 22 can plausibly refill before reset. I’m testing the level-2 pattern exactly on level 3: lower refill → black checkpoint → upper refill on the boundary step → portal → target.
- The refill-boundary route is negative. The target portal itself has another untested context: every prior entry was upward from row 1, which sends the avatar above the sealed chamber. The adjacent black tile provides a route to row 0, column 9, allowing lateral entry into the same colour-1 portal. Given the directional portal behavior already observed at the left gate, I’m testing that side entry and its destination.
- The solved level provides the crucial context: level 2 enters its black L-shaped tile from above with `DOWN`. Every level-3 probe touched the asymmetric black tile from the right with `LEFT`. I’m now reproducing the verified affordance exactly—move through the tile, step above it, re-enter downward—then take the portal and test the sealed target. This still fits comfortably after the last refill.
- The downward black entry alone is insufficient. The remaining exact-budget chain combines both distinct level-3 mechanisms: collect the central key, approach the black keyhole directly from above (without first touching it laterally), then portal and target. From the last refill this is exactly 21 moves, which strongly matches the level’s deliberate energy geometry.
- The key/keyhole chains are negative, so those objects are passive/hazardous rather than unlockers. One mechanical difference remains between solved level 2 and the level-3 door: level 2 approaches the target after a leftward run, then presses down; level 3 arrives facing up from the portal. Blocked lateral inputs can still change facing without moving. I’m testing the compact facing contexts before the down interaction.
- The apparent bottom-right “target” is fully embedded in a solid colour-5 display chamber, so direct movement cannot solve it. I’m running a much smaller observational search now: avatar tile plus remaining refill markers only, ignoring the costly energy/HUD animation. This collapses the maze to a few hundred states and should reveal any shortest reward route based on reachability and pickup state without the previous clone explosion.

## Preserved local probes

- `probe_level.py` (1030 bytes)

Run or inspect the smallest relevant probe before writing another one.
