# Unverified frontier brief: su15 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed API surface, then reproduce any preserved observations before probing level 3. I’ll keep all environment experiments bounded and symbolic.
- The baseline reliably reaches level 3 after 21 moves. There is no preserved frontier brief, so I’m now inspecting only the indexed solver ranges plus the documented harness entry points needed to create a level-3 clone.
- The harness confirms this is a coordinate-action game surface. I’m adding a disposable, compact probe that advances through the two known legs, summarizes level-3 components, and tests both opaque actions on clones at object centers and a few empty cells—without committing probe moves to the real run.
- Single-click deltas show the three playfield color-6 squares are the manipulable objects; empty clicks only touch a one-pixel turn indicator, while clicks near a square change that square locally. The two color-9 rings are stationary targets. I’m now testing the known merge leg on a clone and measuring the resulting object counts/distances as a dense progress signal.
- The failed direct route exposed a real movement constraint: the remaining 3×3 piece advances on a 6-pixel diagonal lattice, and the occupied ring blocks a straight descent. I’m testing two legal zig-zag routes above that barrier, with ring entry from the left versus right edge, before encoding the reusable delivery leg.
- The verified route completes level 3: after reusing the existing merge-and-deliver leg, a general diagonal detour moves the remaining 3×3 piece above the occupied ring, approaches the empty ring from its outside edge, and raises reward from 2 to 3. I’m encoding exactly that reusable skill in `legs.py` and keeping `play_level_3` composition-only.
- The integrated solver now reports `levels=3`, `moves=64`, `replay_ok=True`, with no error. I’m removing the disposable probe file and doing one final replay validation on the clean solver surface.
- Implemented level 3: - Added reusable diagonal ring-detour leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/legs.py:112). - Added composition-only level dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/players.py:13). - Removed the temporary probe. Final validation: `RESULT levels=3 moves=64 replay_ok=True err=None`
