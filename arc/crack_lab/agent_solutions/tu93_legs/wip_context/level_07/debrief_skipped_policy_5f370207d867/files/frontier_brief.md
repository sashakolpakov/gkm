# Unverified frontier brief: tu93 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed signatures/ranges, then reproduce any preserved observations before probing level 6. I’ll keep all experiments bounded and symbolic, and only commit reusable leg composition to the player.
- The clean baseline is confirmed: the current solver reliably reaches level 6 entry (`levels=5`, replay valid). I’m now narrowing the harness surface to the clone/replay entry points and reproducing level-6 action effects from a fresh clone.
- The level-6 frame reproduces a known maze vocabulary in a new configuration: color 9 is the controllable avatar (only action 3 moves it initially), color 14 is the goal, and six color-8 nodes each carry a one-pixel color-15 directional marker. Two color-12 agents advance on successful avatar moves. I’m testing the existing direction-marker replanning leg unchanged before considering any new mechanic.
- The reused marker leg makes real dense progress—four of six directional nodes disappear—but then the avatar becomes trapped in a remaining node (its internal marker changes from color 4 to color 11), so the static planner’s route is unsafe in the turn-driven configuration. I’m replaying those 12 decisions symbolically to locate the first unsafe transition and distinguish a direction-rule issue from an autonomous-agent collision.
- The failure is not the directional mapping: the same top marker is safely entered from the right elsewhere. The fatal node had previously been occupied and restored by an autonomous color-12 agent, revealing hidden interaction state despite an identical visible marker. I’m testing bounded lookahead from the untouched level entry; if the full visible state is sufficient, that gives a minimal reusable dynamic-maze leg without hard-coding this layout.
- The direct bounded search is still running beyond the fast probe window, so I’m letting this single bounded attempt finish while staying within the 20,000-state cap. If it fails, I’ll switch to a denser state target (safe marker clearance) rather than expand the cap.
- A 28-move winning path exists and validates to level 6, proving the visible state is sufficient; the full sparse-reward search took about a minute. Before finalizing, I’m testing the existing dense-progress dynamic leg with color-8 node count, which should be substantially more decisive and better matches the cooperative moving-agent mechanic.
- The dense-progress leg now solves level 6 in about three seconds. It tracks color-15 markers at the original waypoint locations, so autonomous overlap does not look like progress, and only accepts a cleared marker after the color-9 avatar has left that node. I’m doing a final deterministic replay and syntax check with all probe output removed.
- Implemented level 6 using a reusable dynamic directional-waypoint leg. - Added [drive_dynamic_directional_waypoints](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/legs.py:392). - Added the level composition in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/players.py:29). - Syntax checks pass. - Final validation: ```text RESULT levels=6 moves=127 replay_ok=True err=None ```

## Preserved local probes

- `wip_glue_notes.md` (292 bytes)

Run or inspect the smallest relevant probe before writing another one.
