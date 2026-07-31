# Unverified frontier brief: s5i5 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The upstream-joint search also exhausted at 8,000 states, so that hypothesis is falsified. Re-reading the verified control cycle reveals a much smaller route: rotate the lower rail through west → down → north while it is below the wall. That yields the same north-threading geometry with the terminal slider reversed. I’m testing whether this alternate orientation can then extend west into the target.
- The compact-chain rotation is now mapped through all four orientations, but the first link cannot be rotated while extended near the lower wall. The geometry indicates the actual final pose must keep the middle pivot above the terminal joint: I’m running one last bounded structural search for `(middle pivot≈(2,31), terminal joint≈(16,31), marker=(16,25))`, which captures the west-facing requirement without assuming an impossible whole-stack translation.
- The structural search has found the expected west-facing geometry, but only at column `43–44` on the right side of the wall. It is still expanding the bounded frontier for a configuration that threads that pivot through column `31`; no reward path has appeared yet.
- The contextual upper-control replay confirms the isolated control only moves the separate color-8 assembly; the lower links and marker do not change, so cooperation is not the missing mechanic. I’m switching to a macro planner that treats each slider run-to-stop as one transition and each rotation as one transition. This preserves verified physics while collapsing the long repeated-click paths that made the earlier single-step searches too shallow.
- The macro planner is progressing normally within the clone-rate budget. It has not yet improved on marker `(16,31)`, which is expected because the next valid macro must temporarily move the marker away while reorienting the deeper links.
- The contextual control is now conclusively ruled out: after unlocking and rotating color `8`, every lower joint and the lower marker remain unchanged. A more promising verified property is that these are sliders, not fixed-length links. I’m testing whether the south-oriented final rail can slide through its parent pivot into the upper chamber; if it can, its perpendicular extension will point west exactly toward the target.
- The wall layout supplies the missing route: there are two three-cell openings, at columns `30–32` and `6–8`. The terminal’s west-oriented pose is blocked from moving north by the central wall, so it must first translate along the lower corridor toward the left opening. I’m verifying how far the two coupled horizontal sliders can stage it west before the northward extension becomes available.
- The verified maze geometry refines the route: the central lower bar ends at row `38`, so the lower marker must first reach the bottom corridor near `(40,7)`, passing beneath that bar, before it can climb the left opening. I’m using that as the next dense subgoal; it is a concrete reachable waypoint rather than another guess at the final pose.
- The lower waypoint is verified exactly: from the corridor pose, `A>` ×3, `D`, `C>` ×3 places the lower marker at `(40,7)` beneath the central bar. I’m now starting from that reproduced state and optimizing the next dense waypoint `(16,7)` through the left opening.
- The second waypoint is also verified: `H>` ×8 climbs cleanly from `(40,7)` to `(16,7)` through the left opening. The terminal is now above the walls; the remaining step is horizontal translation to `(16,25)`, which should be available through the coupled middle slider. I’m validating that final segment and reward now.
- The direct final translation is blocked, which exposes the recursive structure: the marker crossed the left opening, but its parent joint is still at `(39,32)`. The next subgoal is therefore to stage that parent joint beneath the left opening at `(39,8)`; only then can the terminal assembly be reoriented above the wall and approach the target from the west.
- The parent joint staging is verified too: retract `H` five times, move `F` twice, then alternate `C<`/`F>` six times to place the parent joint at `(39,8)`. The remaining geometry is now local to the left opening: rotate the compact child rail north, extend its joint to row `16`, then extend the terminal east into the target.

## Preserved local probes

- `probe_level2.py` (2922 bytes)
- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `probe_level5.py` (3516 bytes)
- `probe_level6.py` (3311 bytes)
- `probe_level7.py` (5222 bytes)
- `probe_level7_ascii.py` (1273 bytes)
- `probe_level7_choreo.py` (1933 bytes)
- `probe_level7_frontier.py` (1909 bytes)
- `probe_level7_joint.py` (13171 bytes)
- `probe_level7_solution.py` (1592 bytes)
- `probe_level7_stage2.py` (2026 bytes)
- `probe_level7_trace.py` (1402 bytes)
- `probe_level7_turns.py` (2344 bytes)
- `probe_level7_upper.py` (1444 bytes)
- `search_level4.py` (2167 bytes)
- `search_level5.py` (3054 bytes)
- `search_level6.py` (2339 bytes)
- `search_level7.py` (3022 bytes)
- `search_level7_local.py` (3314 bytes)
- `search_level7_macros.py` (3203 bytes)
- `search_level7_pivots.py` (3980 bytes)
- `search_level7_west.py` (3810 bytes)

Run or inspect the smallest relevant probe before writing another one.
