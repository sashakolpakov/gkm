# Unverified frontier brief: sc25 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with the index and frontier brief if present, then reproduce only the observations needed before editing the leg composition.
- The index shows no prior frontier brief and no level-5 player yet. I’m running the harness now to establish the exact starting level, then I’ll inspect only the relevant APIs and existing leg/player ranges.
- Level 5 is reached after the existing dispatcher completes four levels. I’m adding a compact, observational probe that invokes the documented harness, stops at level 5, and reports color/components plus clone deltas for key actions without dumping the grid.
- The clone deltas identify the avatar experimentally: a 4×4 composite at logical position near `(row 16, col 36)`, with actions 1–4 moving it up/down/left/right by one tile. I’ll now test whether navigation alone can trigger reward with a small bounded BFS; this also checks walls and reachability without assuming the visible target is sufficient.
- A single coordinate action only toggles one cell in the bottom 3×3 panel; clicks in the world are inert. The three framed glyphs at upper left encode distinct 3×3 patterns (colors 15, 11, and 6), so I’m testing those patterns as control selections and then repeating movement experiments in each selected context.
- The pattern contexts materially change the controlled body: the corner glyph relocates it into the isolated right-hand corridor, while the vertical glyph leaves it in the central region. This confirms a multi-region control mechanic. I’m checking each region independently for reachable reward and interaction events before planning a cooperative sequence.
- The board now resolves into a cooperative layout: the right-hand runner is trapped below two colored barriers (13, then 12) on the route to the 9/10 goal; matching 13 and 12 switches sit in the central and lower-left regions. I’m using distance-to-switch as the dense progress measure and searching each form’s reachable region for short verified paths.
- Each glyph appears to select a distinct avatar, so I’m verifying persistence directly: move one avatar, switch away, then switch back and observe whether its position is retained. That determines whether the next search must track three cooperating bodies or one transforming body.
- The selector has queued/stateful behavior, so a hand-coded assumption about mode timing would be brittle. I’m switching to a bounded macro search over only the three experimentally verified glyph selections plus four movement keys, deduplicating the observable playfield and preserving the exact primitive replay path.

## Preserved local probes

- `probe_level5.py` (11024 bytes)

Run or inspect the smallest relevant probe before writing another one.
