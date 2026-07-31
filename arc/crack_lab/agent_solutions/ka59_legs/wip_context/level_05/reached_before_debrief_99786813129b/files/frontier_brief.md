# Unverified frontier brief: ka59 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin with the indexed surfaces. I’ll reproduce any inherited observations with bounded clone probes before relying on them, then implement level 4 through reusable legs only.
- Baseline is reproduced: the current solver reliably completes three levels in 94 moves. I’m now inspecting only the small player/leg bodies and the harness’s documented environment entry points needed to reach and clone level 4.
- Level 4 has three movable candidates and three matching hollow targets: two small color-14 rings, one large color-11 ring, with a horizontal barrier splitting the board. Arrow probes identify the initially selected small ring at `(x=21,y=30)`. I’m verifying selection and movement for the other two pieces, then I’ll map barrier-respecting reachability and optimize target distance as the dense progress measure.
- The large ring is not selectable; selecting its center leaves the first small ring active. So level 4 reuses the level-3 physical affordance: a small ring must push the large one. The two small rings are selectable and move three pixels per arrow. I’m using a bounded state search keyed by the visible frame plus the active small-ring position, prioritized by minimum assignment distance to the three target centers; this preserves hidden selection state without inspecting runtime internals.
- The first search revealed another reusable rule and improved the dense metric from 117 to 72: when two small rings touch, one can push the other by a 15-pixel lane step. It also exposed why naive connected components are insufficient—touching same-color rings merge. Their hollow centers remain distinct (`0` for active, `5` for inactive), so I’m switching object tracking and search keys to those observable centers; selection state is visibly encoded after all.
- The corrected tracker is now searching rather than crashing; malformed intermediate visual detections are deprioritized instead of aborting the run. I’m keeping the cap at 20,000 visible states and will replay any candidate from a fresh level-4 clone before editing the solver.

## Preserved local probes

- `probe_l4.py` (3341 bytes)

Run or inspect the smallest relevant probe before writing another one.
