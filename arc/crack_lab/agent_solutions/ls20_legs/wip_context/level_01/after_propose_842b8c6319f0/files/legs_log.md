# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1
ls20 level 1 is a sliding-block maze: a 5x5 tile (colors 12/9) slides exactly
one tile-length per action (1=up,2=down,3=left,4=right) along a color-3
"track" network; a move only succeeds if the whole destination 5x5 is track.
The win condition (a color-5 "room" entrance that looks blocked from one
route) actually opens only after the block has passed over a specific other
track cell first (a hidden switch/key), so naive shortest-path-by-position
search can miss it -- reachability isn't just a function of block position.
`find_winning_path` (BFS deduped on the FULL raw frame, not on any
hand-picked object position) sidesteps this correctly and found the level-1
solution (13 moves) in ~2000 explored frames / ~17s. Lesson for future
levels: prefer full-frame-hash dedup over guessing which object's position
is "the state" -- hidden switches/keys are easy to miss otherwise.
solve_by_search(env) (BFS+replay) solved level 1 with zero level-specific
code in players.py; keep reaching for it first on new levels before writing
anything bespoke, and only fall back to hand-written legs if the state space
is too large for a bounded BFS.
