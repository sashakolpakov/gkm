# Unverified frontier brief: re86 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and the documented harness surface. I’ll begin with `solver_index.md`, then reproduce any useful frontier observations with compact clone probes before editing the level-8 player.
- Baseline is confirmed at `levels=7`; level 8 is the active state. The frontier brief’s claims are still only hypotheses, so I’m reproducing them from the smallest preserved probes and the documented clone/action methods before relying on their routes.
- The compact reproduction confirms the core mechanics: action 5 cycles between two movable outlines; actions 1–4 translate the selected outline on a 3-pixel lattice; bordered obstacles can deform and repaint it. The first candidate route achieves the exact 19×7 target geometry, while naïve descent crosses a conflicting swatch and changes color 11→14. I’m now testing the preserved “first shape already placed” route and searching only the second shape’s observable states against real reward.
- The first-shape route is now reproduced, but it exposes an important correction: it reaches the four 11-marker corners with the required 19×7 outline while remaining color 12. I won’t assume color matching is required until reward proves it. I’m switching from full-frame BFS to a compact shape-state probe to identify exactly which barrier interaction converts a 13×13 square into the second target’s 10×16 outline.
- A six-move deformation for the second outline is now verified: it creates the exact 10×16 perimeter with all 48 cells intact. Translating it straight onto the four 6-markers leaves four cells unresolved and does not complete the level. The earlier search missed this state because it assumed the cursor stayed at an integer geometric center; deformation shifts it to `(50,14)`. I’m correcting the acceptance key and searching locally for a clean placement.
- The corrected local placement search confirms the second outline can visually cover its target perimeter, but only after inheriting color 12 from the first placed outline; reward remains unchanged. That makes color matching a real part of the goal, not decoration. I’ve reproduced a direct state where the first outline is both the required 19×7 geometry and color 11, and I’m now testing whether any barrier crossing can preserve that painted state.
- A key reachability issue explains the failed BFS: its cursor-based key discarded states once the selected outline moved off-screen. The level’s existing mechanics explicitly preserve parked off-screen shapes, so the painted outline can potentially bypass the swatch barrier around a board edge. I’m probing that route now; if it works, it cleanly separates “paint above” from “place below.”

## Preserved local probes

- `probe_l8_contact.py` (1134 bytes)
- `probe_l8_edge.py` (1699 bytes)
- `probe_l8_routes.py` (2741 bytes)
- `probe_l8_symbolic.py` (1643 bytes)
- `probe_level8.py` (3591 bytes)
- `search_l8_color_first.py` (3033 bytes)
- `search_l8_first.py` (2196 bytes)
- `search_l8_geometry.py` (2273 bytes)
- `search_l8_phase.py` (2491 bytes)
- `search_l8_place_second.py` (2689 bytes)
- `search_l8_second.py` (2651 bytes)
- `search_level8.py` (4168 bytes)

Run or inspect the smallest relevant probe before writing another one.
