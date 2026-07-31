# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll begin with the indexed API/signature map, then reproduce only the observations needed on bounded level-6 clones. I’ll keep all inspection and probes inside this workspace and make the player itself a composition of reusable legs.
- Baseline is reproducibly at 5 levels (`moves=111`, replay valid). The frontier is genuinely level 6; I’m now checking the existing reusable-leg interface and the preserved probes’ compact summaries before running any broader search.
- The action model is confirmed independently: clicks switch the selected piece, directions move it by exactly 3 pixels, and `5` is the reward test. The useful new clue is that known wins tolerate motion only away from a piece’s constrained socket axis; this looks like directional projection/visibility rather than exact placement. I’m extracting those constraints from earlier solved boards to turn level 6 into a small combinatorial assignment problem.
- The lower-level wins reveal a consistent geometric rule: pieces form a projected connection network between socket openings and colored terminals; successful placements are bounded by neighboring openings, not by arbitrary absolute coordinates. On level 6, the untouched horizontal marked bar already sits on the right socket and central terminal axis. A previous search omitted that row. I’m testing the resulting central-column assembly before expanding anything else.
- The single central-stack candidate failed, so simple contact is insufficient. The strongest remaining omission is still the horizontal marked bar’s untouched right-socket row: the preserved 6,912-case sweep never tested it. I’m running the corresponding bounded 3,456-arrangement family at the requested rate cap; it reuses the already-validated socket-aligned coordinates for the other three pieces.
- The omitted-row sweep is still progressing normally and has not produced a reward hit yet. Its silence is expected—the probe reports only a win or exhaustion—and the action throttle remains capped.
- Level 1 makes the geometry explicit: the movable bar wins only when its projection forms one continuous connector from the colored source to both socket mouths. Applying that same dense measure in both axes, level 6 has a natural gap-free chain that prior sweeps also omitted: the marked horizontal bar at row 35, with the vertical bar ending directly above it and the lower L beginning directly below it. I’m testing that exact assembly now.
- The polyomino orientations resolve the missing placement more sharply. On the logical 3-pixel grid, the upper L turns the top source into the upper-left socket at `(row 7, col 9)`, the four-cell vertical bar spans exactly from that turn to the right-socket row, the marked horizontal bar already occupies that row, and the mirrored L turns the lower-left socket toward the bottom socket. The crucial vertical-bar column (`left=32`) was absent from every preserved sweep.
- A second interpretation now fits the lower levels quantitatively: the colored boundary cell is a directional source, and the movable shapes sit on the source-to-socket rays. On level 6 the initial vertical and marked bars already lie on the right and bottom rays; only the two leftward rays need their L-shaped blockers. I’m testing that 54-state, one-piece-per-ray family with the untouched bars preserved.
- The level-1 exhaustive map confirms the key invariant: depth is irrelevant; only the one-dimensional projection matters, and the winning bar is the unique interval whose endpoints meet the socket mouths. For level 6 I’m therefore snapping every piece endpoint to the shared top/bottom port column while holding the most natural side-socket row embedding. That is only 1,029 exact projection arrangements.

## Preserved local probes

- `probe_axis_ranges.py` (1883 bytes)
- `probe_constraints.py` (4143 bytes)
- `probe_l1_map.py` (979 bytes)
- `probe_l6.py` (9720 bytes)
- `probe_search_l6.py` (4397 bytes)
- `probe_snapshots.py` (1466 bytes)

Run or inspect the smallest relevant probe before writing another one.
