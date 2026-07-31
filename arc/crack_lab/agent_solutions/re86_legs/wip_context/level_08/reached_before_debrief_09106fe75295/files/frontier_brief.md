# Unverified frontier brief: re86 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The local corridor search found no preserved crossing from the `22×4` state. I’m mapping the remaining deformation affordance directly: repeated pushes against the bottom and right boundaries. The second target’s verified `10×16` outline is created there, so boundary compression may supply the horizontal/vertical orientations needed to traverse the two perpendicular swatch walls.
- The deformation source is now identified cleanly: the two solid color-1 blocks, especially the lower block at rows 55–59, columns 34–38. The `22×4` outline is pinned immediately to its left. I’m sliding that pinned outline vertically and pushing again; changing which cells contact the block is the direct way to compress it further without involving any paint station.
- The two color-1 blocks and perpendicular swatch walls now form a consistent route: compress to a short horizontal outline at the lower block, pass the vertical swatch column through its 8-cell gap, paint on the left, return through that gap, rotate at the upper-right block into a narrow vertical outline, then descend through the right edge. I’m verifying the first primitive—the expected `7×19` horizontal compression—with one extra push on the known `10×16` route.
- The transport sequence works through the horizontal wall and reaches color 11, but the return crosses the vertical station border at row 12 and repaints to 9. The safe gap is rows 13–20, requiring the horizontal outline’s center at row 16 rather than 15. I’m enumerating only the lower-block compression counts to find the same `7×19` orientation with that one-cell phase shift.
- The painted horizontal outline also cannot cross because its center remains on a 3-cell phase that touches one station border on every attempt. I’m mutating the nine-move lower-block compression route itself; a one-action collision change can shift the cursor relative to the outline, which is exactly what the observed deformations already do. The acceptance condition is only `7×19` with a new row phase.
- A stronger compression is available one step beyond `7×19`: the observed next state is `4×22` while still pinned to the lower block. Once pulled clear, its four-row height should traverse the vertical gap with margin, eliminating the phase problem. I’m swapping that into the same paint-and-return route now.
- The key transport is now verified end-to-end through both swatch walls: a `4×22` outline is painted 11 on the left and returns to the upper-right block still color 11. That block reverses the compression sequence (`4×22 → 7×19 → 10×16 → 13×13 …`). I’m adjusting downward between pushes so the growing vertical outline stays on-screen while remaining pinned, aiming for the reusable `22×4` form.
- The failed descent was a pinning issue, not a color-route failure: while the `22×4` outline overlaps the upper block, right actions cannot move it. Pulling it down two lattice steps clears the block; then three right actions should place its visible columns at 61–63, entirely outside the final station’s paint interior (58–60). I’m testing that corrected release-and-edge sequence.
- The upper block’s minimum is `22×4`; further pushes are correctly blocked. That is still sufficient: the horizontal wall has a six-column opening at 44–49. After clearing the block, moving the narrow outline to columns 45–48 places it wholly inside that opening. I’m testing this interior gap now, which avoids all boundary clipping.
- The full color-11 transport is now verified: the `22×4` outline crosses the lower wall through columns 45–48 and remains color 11. Below the wall, the lower color-1 block is immediately to its left and can reverse the compression. I’m probing left pushes there to recover the `19×7` target orientation, then it can translate to the four 11 markers.
- The lower block does not expand the outline by horizontal reversal; deformation is directional. To widen it, the outline must approach the block’s horizontal face: move above it, align columns 33–36 with the block, then push downward. That should compress height and expand width in the same `22×4 → 19×7 → …` sequence.
- The first complete transport-and-placement route is now determined symbolically: it restores a color-11 `19×7` outline at bbox `(39,9)–(57,15)`. The same pipeline can paint the other mover color 6 and expand it to the `10×16` target. I’m running the composed clone route now; the decisive check is whether the engine preserves per-object colors at the four geometric crossings or forces an additional weaving deformation.

## Preserved local probes

- `probe_l8_bypass.py` (1670 bytes)
- `probe_l8_contact.py` (1134 bytes)
- `probe_l8_edge.py` (1699 bytes)
- `probe_l8_relay.py` (1636 bytes)
- `probe_l8_routes.py` (2741 bytes)
- `probe_l8_squeeze.py` (2438 bytes)
- `probe_l8_symbolic.py` (1643 bytes)
- `probe_l8_transport.py` (2536 bytes)
- `probe_level8.py` (3824 bytes)
- `search_l8_beam.py` (2528 bytes)
- `search_l8_color_first.py` (3033 bytes)
- `search_l8_first.py` (2196 bytes)
- `search_l8_geometry.py` (2273 bytes)
- `search_l8_mutations.py` (1893 bytes)
- `search_l8_orientations.py` (2548 bytes)
- `search_l8_phase.py` (2491 bytes)
- `search_l8_place_second.py` (2689 bytes)
- `search_l8_second.py` (2651 bytes)
- `search_level8.py` (4168 bytes)

Run or inspect the smallest relevant probe before writing another one.
