# Unverified frontier brief: dc22 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin from `solver_index.md`, then reproduce any prior observations before relying on them. I’ll use compact clone probes and keep all level logic in reusable legs.
- The baseline is still starting after a one-time plotting cache build; no prior `frontier_brief.md` exists, so I’m establishing the level-6 observations fresh.
- The baseline confirms the solver stops exactly after five completed levels. I’m now capturing only compact level-6 symbols—action set, color counts, connected objects, and one-step deltas—to identify the avatar and candidate controls without dumping the grid.
- The avatar is now experimentally identified: the 2×2 color-14 object at rows 52–53, columns 28–29; action 1 moves it up one tile, action 3 left, while 2/4 are blocked initially. The only active coordinate control is the top-right object centered at `(56, 8)`, and it translates a lower movable assembly. Next I’m reducing the playfield to its aligned 2×2 tile map and testing that control across repeated phases.
- The control has a six-phase cycle. Phases 4–5 enlarge the avatar’s reachable set from 18 tiles to about 30, proving the moving color-1 assembly is a traversable bridge/platform. A static click-then-walk plan still cannot win, so the next probe interleaves platform phases with movement, using the observed reachable-area expansion as the dense progress signal.
- The interleaved search is slower than the static probes because clone depth grows with each platform phase. It is bounded at 4,000 observational states; I’m letting this first run finish, then I’ll shrink the state key if it confirms the expected bottleneck.
- The replay-based search avoids recursive clone growth, but reconstructing every candidate path is still costly. If this bound does not finish promptly, I’ll switch to targeted platform-transfer probes around the newly reachable left island instead of broad exploration.
- The dense transfer target succeeded. A 21-action interleaving carries the avatar onto the lower-left island, which activates a second control at `(51, 25)`. That control reconfigures the large color-12 central assembly; I’m now using the same verified transfer method to reach the next previously disconnected island and discover the following active control.
- The second assembly can immediately extend one tile above the lower-left island, but the next fixed island is much farther up. I’m bounding the search by avatar position plus the two observed assembly shapes; the current run is testing whether the two existing controls suffice for that full crossing.
- A single activation of `(51, 25)` connects the lower-left island all the way to the upper-left island; fifteen upward steps reach it. No new coordinate control appears merely on arrival, so I’m testing the distinct color-10 tile on that island as the likely endpoint trigger, matching the earlier level pattern where walking onto a special tile reveals the next control.

## Preserved local probes

- `probe_level6.py` (6750 bytes)

Run or inspect the smallest relevant probe before writing another one.
