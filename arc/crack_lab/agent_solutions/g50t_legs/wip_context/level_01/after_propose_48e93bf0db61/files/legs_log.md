# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## g50t level 1 (fingerprint 14872b062f27)

Discovered mechanics (raw-frame experiments on clones only):
- Colours: 0=background/walls, 5=floor, 9=avatar+goal+HUD, 8=a rigid "loop"
  obstacle, 1=HUD/step-counter fill, 2=appears only via USE (a phase marker).
- Avatar = the hollow 5x5 ring of colour 9 at top-left of the maze. Actions
  1/2/3/4 = up/down/left/right, translating the avatar by a fixed pitch of 6.
- Action 5 (USE): toggles a 2-phase HUD, resets the avatar to its home cell,
  and (in phase B) stamps a static colour-2 anchor ring near home. The anchor
  never moves and does not open anything.
- Action 6 (coordinate) is inert here (no cell responds; step(6) just wastes a
  turn). env.actions=(1..5); step accepts 1..6.
- Bottom border row is a step/timer counter (9->1, one cell per ~2 moves).
  It only ever depletes; exhausting it => terminal LOSS (levels_completed=0).
  Move budget ~= 129 moves.
- The only object interaction: when the avatar reaches the top-right tile
  (8,38) it PUSHES the colour-8 loop, which slides so the plug blocking the
  left col-14 corridor moves aside and the corridor opens top-to-bottom.
  BUT this "open" state exists ONLY while the avatar occupies (8,38); moving
  off it (the only non-USE exit is LEFT) reverts the loop immediately.

Reachability / solvability (proven, not guessed):
- Complete clone game-tree enumeration (full 64x64 frame keys, all actions,
  no dedup across the counter) = exactly 1854 reachable states, max depth 129,
  40 terminal states — ALL with levels_completed=0.
- Only 2 distinct colour-8 configurations exist across the entire tree, and the
  goal chamber (bottom-right) and the whole lower third of the board are NEVER
  reached or modified. Avatar's lowest reachable row is 32; the goal is ~row 50.
- Verified clone==real (frames, terminal, counter all match; clones isolated).
  Replayed all 25 distinct world-configs on the REAL env: every one gives
  levels_completed=0. Dwelling to timeout in any config => loss.

Conclusion: this start is a single-agent "hold-the-switch" impossibility — the
only barrier (the col-14 plug) requires a body to remain on the (8,38) switch
while another body descends col-14, but there is exactly one controllable body
(the colour-2 anchor is static and cannot be steered onto the switch or through
the corridor). No env.step sequence increases levels_completed. Consistent with
the pre-recorded failure "1:14872b062f27".

Leg added: `solve_level_by_search` (+ `bfs_to_goal`, `state_key`) — a general,
game-agnostic bounded clone-BFS that commits any found level-completing path.
It is correct and reusable; it finds nothing here only because the tree has no
winning state. Reuse it for any genuinely solvable maze/switch configuration.
