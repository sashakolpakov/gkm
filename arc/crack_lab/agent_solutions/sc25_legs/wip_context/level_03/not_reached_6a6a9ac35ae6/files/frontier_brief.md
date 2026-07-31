# Unverified frontier brief: sc25 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the documented harness/perception surface. I’ll begin with the indexed source map, reproduce any carried-forward observations, then probe level 3 symbolically before editing the leg library and player dispatch.
- The baseline is confirmed at level 3’s entry: `levels=2`, 22 moves, replay valid. I’ll now reproduce the two potentially decisive mechanics from the brief—coordinate-grid transformation and the push/socket interaction—using the smallest preserved probe and fresh level-3 clones.
- The reproduced frame corrects an important ambiguity: the move-budget bar is in the final two columns, not the final two rows. The carried-forward search key therefore retained elapsed-turn noise and could miss the true finite movement state space. I’m rerunning bounded movement search with the bar excluded correctly; this directly tests whether the verified grid leg plus movement is sufficient.
- The board has a visible 32-input budget bar, so excluding it is valid for spatial reachability but not for delayed outcomes. I’m separating those hypotheses now: direct long replays test whether a correctly prepared board completes only when the budget expires, while preserving the spatial result that no gate is reachable by movement alone.
- The harness is resuming from its validated checkpoint, so level-1/2 probe hooks never execute. I’ll temporarily move that documented checkpoint aside within the workspace for one fresh replay, restore it automatically, and use only the reward-transition summaries—no checkpoint contents or prior artifact inspection.
- Known reward frames identify the bottom 9/10 shape as the exit: earlier levels complete when the avatar contacts that same shape. Level 3’s dense subgoal is therefore opening the color-13 barrier between the reachable floor and that exit. The remaining untested action affordance is bare action `6` (distinct from coordinate clicks), so I’m probing it at the barrier and right-side device before assuming the device is inert.
- Partial-grid probes still move only the identified avatar, so there is no control handoff. I’m now exhausting the actual timed movement state space with the full frame retained (including the budget bar); its upper bound is modest, and this will catch any visit-order or delayed gate behavior that a purely spatial key intentionally collapses.
- Movement cannot solve the prepared board in any timed state. The coordinate panel is fully toggleable, so I’m using a bounded 512-configuration experiment: for each visible 3×3 pattern, route the avatar to the barrier and attempt the short crossing-to-exit sequence. Reward is the real goal signal; barrier displacement or avatar passage is the dense intermediate signal.
- The color linkage suggests a dependency chain: the 3×3 clue contains color-6 marks, the right object has a color-6 shell around a color-13 core, and color 13 blocks the matching avatar exit. The clue itself has not yet been used as a coordinate target. I’m probing its three color-6 marks and the ring (including latent control changes), then I’ll test the barrier route if any handoff occurs.

## Preserved local probes

- `probe_level2.py` (7657 bytes)
- `probe_level2_interactions.py` (4048 bytes)
- `probe_level3.py` (4545 bytes)

Run or inspect the smallest relevant probe before writing another one.
