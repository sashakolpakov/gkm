# Unverified frontier brief: sk48 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surface. I’ll inspect only the listed ranges, then reproduce any inherited observations with bounded symbolic clone probes before editing the leg library and level-4 player.
- The baseline reproduces cleanly: 3 levels in 133 moves, with replay validation passing. I’m now reading the existing leg conventions plus the minimal harness/API ranges needed to stop on a level-4 clone; then I’ll identify the avatar and action semantics afresh on that clone.
- The level-4 start is now experimentally characterized. Action 1 moves the avatar and its attached four-token train up one six-pixel lane; action 2 is boundary-blocked; actions 3/4 retract and extend the tether; bare 6 is inert. The physical train is `9,14,8,12`, while the separated lower region shows `8,12,9,14`, so my dense measure is preservation of the ordered suffix `8,12` while staging `9,14` for later append. This is the same family as the existing row-staging leg, so I’m testing that reusable structure before adding any new mechanic.
- The first staging template did not alter token order, so I rejected it rather than forcing a level-specific parameter guess. A new verified constraint explains why: the whole train can traverse all six vertical lanes unchanged, so vertical motion alone cannot reorder it. I’m switching to a compact physical-state search keyed only by avatar/tether geometry and token positions, with the dense goal “all four tokens form `8,12,9,14` left-to-right”; this removes static maze pixels and keeps the probe bounded.
- Coordinate action 6 is inert on every observed object center at the start, and coordinate 7 is rejected by the documented environment; that rules out selection/click mechanics here. The full-order search was too coarse, so I’m decomposing it further: first find the shortest verified state where the four tokens occupy different rows. That directly identifies how this maze lets the tether separate a prefix/suffix, and should yield a reusable weave-style leg instead of a blind long search.

## Preserved local probes

- `probe_level2.py` (2280 bytes)
- `probe_level3.py` (5753 bytes)
- `probe_level4.py` (4414 bytes)
- `wip_glue_notes.md` (293 bytes)

Run or inspect the smallest relevant probe before writing another one.
