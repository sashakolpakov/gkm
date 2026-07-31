# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Verified level-9 mechanics

- The local board contains four reusable color-9 bridges and two pegs.
- Any movable bridge may jump another occupied piece without consuming it.
- A peg crossing a bridge preserves the bridge; a peg crossing a peg captures it.
- One remaining peg is not sufficient: it must enter the bordered carrier to reveal the wrapped relay.
- The shortest observed local solution is 14 jumps / 28 actions.
- The wrapped relay has a verified 74-action solution. Its far bridge pair must be relayed to the remote peg; direct long jumps, edge-wrap clicks, vertical carrier turns, selected-key moves, and transition undo cycles were falsified on pristine clones.
- The composed 102-action level leg reaches `levels_completed=9` on a clone.

## Preserved candidate and cap blocker

- `level9_full_candidate.json` is independently replay-valid for 9 levels at 646 total actions.
- Normal `python gkm_try.py` stops at 600 with level 8 because the supervisor checkpoint prefix is still 544 actions.
- Prefix loop erasure, exact key-run shortening on levels 5 and 7, suffix-aware shortening of the first five level-7 runs, all 23 individual level-7 coordinate-pair removals, and bounded level-4/5 macro beams produced no saving.
