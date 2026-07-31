# Unverified frontier brief: s5i5 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin from the indexed API surface. I’ll reproduce any inherited observations with bounded clone probes, then implement level 3 through reusable legs only.
- The baseline reliably reaches level 3 in 39 moves. I’m now inspecting only the indexed function ranges and the harness entry points needed to construct a level-3 clone, then I’ll summarize the raw frame into components rather than dump pixels.
- Level 3 is a coordinate-only control puzzle: the upper playfield contains several colored articulated shapes, while the lower panel has six paired controls. I’m testing each control half on independent clones and measuring which upper component moves, its displacement, and whether repeated clicks saturate or reverse.
- The reward structure is now identifiable: two colored slider tips each contain a single color-13 marker, and the central color-15 object contains two hollow cells surrounded by color 13. Dense progress is the sum of the two marker-to-socket distances. Each direct route is initially blocked by two perpendicular articulated bars, so I’m probing staged blocker displacement before committing a reusable solver.
- A key correction from the collision probes: the short color-7 and color-10 sliders are meant to make contact with the articulated bars; their apparent “blockage” is a verified attachment subgoal, not an obstacle-clearing failure. I’m using a bounded reward search over the twelve documented coordinate controls to identify the remaining attachment order, then I’ll generalize that sequence from perceived control effects.
- The direct-clone breadth search is too expensive for this runtime and has not yielded within its bounded window, so I’m stopping it rather than spending the move budget blindly. I’ll switch to compact state/replay probes around the verified attachment states.
- The compact best-first probe confirms the dense metric is meaningful: it reduced the total socket distance from 60 to 15 and found states where one tip is effectively routed, but the initial depth cap is too tight for both. I’m preserving the best symbolic path and extending only around that frontier rather than widening the search.
- The bounded search found and verified a 51-click win. The successful mechanic is cooperative staging: partially extend both goal sliders, retract and reposition the two articulated joints to pass each blocker, dock the horizontal tip, then lower the vertical tip into the second socket. I’m encoding that as one reusable leg with controls discovered from the lower-panel components; the level player will contain only that leg call.

## Preserved local probes

- `probe_level2.py` (2922 bytes)
- `probe_level3.py` (2988 bytes)

Run or inspect the smallest relevant probe before writing another one.
