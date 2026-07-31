# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and the documented harness surface. I’ll begin with `solver_index.md`, then read `frontier_brief.md` if present, and keep all level-7 experiments clone-based until the composed leg is verified.
- Baseline is confirmed: the current solver reaches level 7 with `levels=6`, 196 moves, and a valid replay. I’ll now reproduce the preserved observations using the smallest existing probe, then inspect only the narrow harness and leg ranges needed to identify a bounded level-7 plan.
- The preserved probe exposed one stale assumption cleanly: the current search has no `support_radius` or `edge_gravity` options, so its claimed result was never reproduced. The level-7 frame itself is reproduced: the avatar begins at lattice `(6,1)`, with shape-changing supports at `(6,4)` and `(8,4)`, hazards at `(4,1)` and `(4,5)`, and a row of color-8 controls along the top edge. I’m separating interaction reach from traversal now, using short paths only.
- The short probes establish the key affordance precisely: a color-12 support changes from area 5 to 21 only when the avatar is close enough horizontally; the missing useful case is exactly two lattice rows away vertically. I’ve generalized the existing gravity-room leg with a configurable support radius (default unchanged for levels 4–5) and am running the bounded level-7 search with radius 2.
- The control ambiguity is now resolved experimentally. Only the leftmost color-8 object at pixel `(3,3)` changes gravity; the apparent “controls” across the top are overlays on ordinary color-12 supports, and clicking their centers only expands those supports. This is why the existing lattice-only search never generated a gravity action. I’m adding one optional edge-control action to the same reusable search and will verify the resulting route before wiring level 7.
- The generalized search is still within its 5,000-state cap, but reconstruction makes this pass slower than the desired steady-state budget. If it does not produce a route at this cap, I’ll switch to a compact observational BFS keyed without the move-counter strip, retaining the same experimentally justified action set.
- The first complete macro is now verified and yields a real dense-progress signal: it shifts the entire vertical control strip down exactly one 6-pixel band while keeping the avatar alive. The slow search was spending its budget deep-copying replay histories, so I stopped it. I’m treating that band shift as the subgoal and testing whether the same support–flip–cross pattern can be repeated room by room.
- The useful staging sequence is now reproduced: align the avatar with the lower support, expand it, flip gravity, step across the upper wall, flip again, then step right. That safely scrolls into the next room; taking the same right step without the staged landing is terminal. I’m using this verified eight-action opener to cut the search frontier down to the genuinely new room.

## Preserved local probes

- `probe_level7.py` (5105 bytes)
- `probe_level7_controls.py` (1908 bytes)
- `probe_level7_macro.py` (2995 bytes)
- `probe_level7_room2.py` (1455 bytes)
- `probe_level7_search.py` (1024 bytes)
- `probe_level7_stages.py` (5000 bytes)

Run or inspect the smallest relevant probe before writing another one.
