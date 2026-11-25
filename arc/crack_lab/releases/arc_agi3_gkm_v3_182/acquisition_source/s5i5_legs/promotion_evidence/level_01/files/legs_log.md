# Verified level 1 observations

Reproduction scripts: probe_initial.py, probe_clicks.py, probe_align.py.
All experiments use the public frame, actions, clone and step surface.

- Only advertised action is coordinate click 6. Background clicks change only the bottom move indicator; no autonomous bar motion was observed over four such clicks.
- The horizontal bar has a pink tip initially at (31,10), and a target cross centred at (52,10). Clicking (39,21) retracts its tip three pixels; (45,21) extends it three pixels. Extension then retraction restores the original bar.
- The vertical bar tip starts at (10,34), with a target at (10,52). Clicking (24,38) retracts three pixels; (24,44) extends three pixels.
- Seven horizontal extensions align that bar without awarding a level. Six subsequent vertical extensions award level 1. The horizontal tip approaches the target with a shorter final increment.
- Dense progress: summed Manhattan distance between target centres and moving pink tip markers, excluding the four static pixels in each cross.

Library leg `align_marked_bar_tips` locates crosses and control interiors from the frame, then tests each control on a clone and commits one that reduces distance (or completes the level). `players.play_level_1` only composes that leg.

Validation: `python gkm_try.py` returned `RESULT levels=1 moves=13 replay_ok=True err=None`.
The independently reproduced fixed path is saved as `level1_aligned_bars_path.json`. No checkpoint was edited.
