# Unverified frontier brief: cn04 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API surface, then reproduce any frontier observations with compact clone probes before changing the level-6 solver.
- The baseline run is still starting up because Matplotlib is building a temporary font cache. Once it returns, I’ll inspect only the small level-6 probe and the indexed player/leg ranges needed to reproduce the mechanics.
- The mechanics reproduce cleanly: actions 1–4 move the selected multicolour figure, 5 rotates it, and action 6 at a figure coordinate changes selection. The three green cells are part of the selected figure—not separate targets—and the odd 32-pixel top bar is not built on the same 3×3 logical grid as the pieces. I’m checking a known pre-win transition now to identify the actual completion test before searching level 6.
- The known level-5 win gives the key correction: its four figures contain 44 logical cells but the solved pre-transition frame contains only 38 occupied cells—exactly six cells of intentional overlap. So this is registration/overlay, not edge-joining. I’m now exposing each level-6 figure’s visible colour landmarks when selected; those landmarks should define the reusable alignment skill and a much smaller search space.
- The colour landmarks form a clean exact pairing: all 12 green cells can be paired once across the five figures, yielding a six-edge connector graph. I corrected the rotation model so landmarks and bodies share the same pivot, and I’m replay-testing that unique compact registration now; the test includes safe staging moves before rotating the two edge-adjacent figures.

## Preserved local probes

- `probe_branch.py` (2446 bytes)
- `probe_candidate.py` (1991 bytes)
- `probe_clean_connect.py` (2832 bytes)
- `probe_connect.py` (2560 bytes)
- `probe_final_bfs.py` (978 bytes)
- `probe_finish.py` (1946 bytes)
- `probe_goal.py` (981 bytes)
- `probe_known_shapes.py` (1625 bytes)
- `probe_l6.py` (2302 bytes)
- `probe_layout_search.py` (3780 bytes)
- `probe_piece_colors.py` (2401 bytes)
- `probe_prior.py` (1539 bytes)
- `probe_search.py` (2359 bytes)
- `probe_selections.py` (1953 bytes)
- `wip_glue_notes.md` (1172 bytes)

Run or inspect the smallest relevant probe before writing another one.
