# Unverified frontier brief: g50t level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surfaces. I’ll reproduce any prior observations I use, probe level 6 symbolically on bounded clones, then keep `play_level_6` composition-only.
- Baseline is reproduced: levels 1–5 complete in 203 moves, and level 6 is the first missing dispatch. The brief’s mover claim is still untrusted; I’m rerunning its compact probe and inspecting only the relevant indexed definitions before relying on it.
- The reproduced deltas show two genuinely different actors: the color-9 5×5 mover obeys the chosen direction, while color 14 advances autonomously on every successful turn. Both travel on the color-5 track rather than the zero background. I’m now testing the actual reward condition with a bounded observational BFS; the dense state is the pair of mover positions, not raw “amount changed.”
- The compact pair-position key exhausted without reward, which is evidence that the level carries additional progress state (track consumption or interaction state). The fuller frame-state search is substantially more expensive, so I’m bounding it tightly and will switch to a replay-based search if it does not finish promptly.
- The existing unlock planners do not fit this configuration. A crucial contextual effect is now clear: `USE` after movement resets both racers and advances the three-slot header, so it ends a round; it is not a gate toggle here. The player’s lower track has two terminal branches, while the autonomous racer follows the upper track. I’m exhaustively testing the small set of round outcomes (three rounds, two meaningful endpoints) rather than expanding pixel states.
- The player’s complete movement component has only 17 positions and no direct winning tile. That confirms cooperation/state staging is required. The earlier compact BFS accidentally discarded the three small header markers, so all `USE` phases collapsed together; I’m correcting the observational key to retain those markers while still excluding the irrelevant shrinking timer bar.
- The autonomous object is not merely clock-driven: with a left/right oscillation it loops through a six-state spur, and identical frames can encode different travel headings. That explains why frame- or position-deduplicated searches were incomplete. I’m adding the minimal missing state—its previous position—to the joint reachability probe, while leaving countdown pixels out of the dense measure.
- The joint graph now exhausts at only 321 observable heading/special/header states, so missing navigation is no longer the issue. The remaining changing variable is the bottom color-9 meter, which shrinks monotonically during successful turns. I’m testing whether that meter is a completion clock/progress counter; if so, deduplicating repeated joint states was exactly the wrong abstraction.

## Preserved local probes

- `probe_bfs6.py` (4088 bytes)
- `probe_level6.py` (3411 bytes)
- `wip_glue_notes.md` (1465 bytes)

Run or inspect the smallest relevant probe before writing another one.
