# Unverified frontier brief: s5i5 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with the indexed signatures, then reproduce any prior observations I actually rely on through bounded clone probes. I’ll keep level-specific orchestration in `players.py` and mechanics in reusable legs.
- Baseline is confirmed at level 3 with replay integrity intact. I’m now reproducing the level-4 state from a fresh environment and reducing it to color components, controls, and per-click deltas—no full-frame dumps.
- The level has five paired coordinate controls and five persistent upper-field objects. Independent clicks already show directional saturation and context-sensitive movement; I’m tightening the probe to track whole mixed-color objects (not individual same-color fragments) and to measure each control’s reachable trajectory and distance to the five color-13 cells.
- A bounded clone search found a 30-click win and a clear dense-progress story: the marker advances from row 43 to the hollow target at row 10; the shared extension first meets a long crossbar, then three shorter crossbars. Retracting the long one and ratcheting the three short ones upward opens each successive passage. I’m encoding that reusable “shared extension through staged crossbars” mechanic and then validating it from a fresh run.
- Implemented and verified level 4. - Added reusable `extend_shared_marker_through_staged_crossbars` leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_budgeted_medium_screen/legs.py). - Added the level-4 dispatcher in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_budgeted_medium_screen/players.py). - Final validation: `RESULT levels=4 moves=120 replay_ok=True err=None`.

## Preserved local probes

- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `search_level4.py` (2167 bytes)

Run or inspect the smallest relevant probe before writing another one.
