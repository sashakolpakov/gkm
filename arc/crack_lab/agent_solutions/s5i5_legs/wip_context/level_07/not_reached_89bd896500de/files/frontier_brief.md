# Unverified frontier brief: s5i5 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace boundary, begin from `solver_index.md`, then reproduce any prior observations through bounded clone probes before changing the level-7 player.
- Baseline is `levels=6`, matching the frontier brief. I’m now reproducing the transition to level 7 and will use the existing compact probe style only as scaffolding; no unverified level-7 claim will be carried forward.
- The reproduced post-completion frame is still the solved level-6 board, not the stated seven-colour level-7 substrate. I’m checking the documented step semantics now—specifically whether the first subsequent action advances the rendered board—before interpreting any objects.
- The discrepancy comes from startup state: `gkm_try.py` reaches the frontier through its validated replay path, while my first probe reconstructed earlier levels through the current leg implementations. I’ll align the probe with the harness’s documented replay route and treat only that resulting frame as level 7.
- Level 7 is a recombination of earlier articulated mechanisms: nine control panels affect three separated assemblies, and the only attached colour-13 marker starts near `(7,22)` while the hollow four-cell target is centered near `(16,25)`. I’m using marker-to-target distance as dense progress and will first map each panel’s repeated trajectory, including blocked directions.
- The bounded search has reproduced monotonic progress from distance `30 → 18`: the lower paired controls can move the articulated marker left to `(16,43)`, where the central wall blocks it. The remaining search is exploring only distinct rendered states to find the necessary reorientation around that wall.
- The breadth-first pass is clone-limited and has not found a closer state beyond the wall yet. I’m letting this bounded 30k-state pass finish; if it exhausts without reward, I’ll switch to a denser best-first ordering rather than expanding the cap.
- A direct geometric route works through the first corner: down to `(31,55)`, left to `(31,31)`. Upward motion is then blocked even though the marker itself has clearance, so a connected arm segment is colliding with the vertical wall. I’m probing rotations at that exact corner to identify the minimal fold needed before ascent.

## Preserved local probes

- `probe_level2.py` (2922 bytes)
- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `probe_level5.py` (3516 bytes)
- `probe_level6.py` (3311 bytes)
- `probe_level7.py` (5222 bytes)
- `search_level4.py` (2167 bytes)
- `search_level5.py` (3054 bytes)
- `search_level6.py` (2339 bytes)
- `search_level7.py` (3022 bytes)

Run or inspect the smallest relevant probe before writing another one.
