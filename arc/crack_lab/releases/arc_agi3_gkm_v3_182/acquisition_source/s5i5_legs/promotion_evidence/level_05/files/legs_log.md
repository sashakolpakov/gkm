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

# Verified level 2 observations

Reproduction: `probe_level2.py` and `probe_route.py`, using a pristine entry
obtained by running the level 1 leg, followed only by clone experiments.

- Paired controls extend/retract four connected arm sections. Extension clicks
  at x=12,27,42,57 (y=57) respectively shift the tip right/up/right/down by 3.
- At entry the tip is (16,37); the cross centre is (52,31).
- Visible walls occupy x=33..56 at y=9..11, x=33..35 at y=12..26,
  and x=42..44 at y=15..44. A horizontal passage runs at y=12..14.
- The original alignment leg stalls with its tip at (40,31), reward still 1.
- A clone route of 8 first-section extensions, 8 second-section extensions,
  4 third-section extensions, and 6 fourth-section extensions reaches reward 2.
  Intermediate tips are (40,37), (40,13), and (52,13). Each expected step
  changes the frame; reward remains 1 until the last stage completes.

`route_telescoping_arm_through_gap` derives lattice spacing, controls, target,
arm joints, and clear bend candidates from the frame. It verifies a staged
route on a clone before committing. `play_level_2` only composes this leg.

Validation: `python gkm_try.py` returned
`RESULT levels=2 moves=39 replay_ok=True err=None`.
An independent run from level 1 also reached 2 in 39 moves and validated True.
Its public action path is `arm_gap_level2_candidate_path.json`.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_arc_agi3_n0_fresh_frontier_nxg8j0d5/candidate_coupled_parking_level5.json` and installed a thin replay player for the recovered suffix.
