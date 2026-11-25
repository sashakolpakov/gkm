Level 1 clean observations (reproduced by local probe.py and harness):
- Public actions are (3, 4, 6, 7); all coordinate probes used in-frame integer triples.
- On pristine clones, 3 moves colour 9/11 avatar from x21 to x15;
  4 moves it to x27. Repeated 3 stops at the left boundary.
- Clicking yellow centres (15,3), (21,3), or (33,15) removes the
  corresponding 14-colour tile. Clicking empty space leaves geometry unchanged.
- After four right moves the avatar enters the gap at x45 and the scene
  scrolls down three tile rows while the avatar stays near y39.
- Clicking an overhead yellow tile causes a further rise. Indestructible
  colour 5 tiles require moving sideways to an opening.
- Action 7 after movement or a tile removal restores the preceding geometry.
- A 15-action clone route reached the green tile (colour 7) and increased
  levels_completed from 0 to 1; nearby air and intermediate removals did not.
- candidate_ascent.json preserves that clone path. probe.py reproduces the
  route from pristine entry using only public frames, clones, and actions.

Reusable leg: ascend_breakable_shaft uses a visible tile graph, assigning cost
for horizontal moves and yellow removals. Upward reach is the dense objective;
when visible, the green tile becomes the target. It replans after each step.

Validation: python gkm_try.py
RESULT levels=1 moves=15 replay_ok=True err=None
