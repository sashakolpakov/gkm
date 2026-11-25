# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 2: align_indicator_with_transfers

Reproduced from pristine level entry in probe_level2.py and probe_transfer.py.
Only advertised action is coordinate ACTION6. Four color-9 button centers,
ordered vertically, are (x,y)=(1,17),(1,25),(1,37),(1,45). Initial region
widths sampled at rows 8,30,60 are 52,12,8. Button 2 transfers four pixels
from top to middle; button 4 transfers four from middle to bottom and moves
the six-high color-14 marker right four pixels. Button 4 stalls when middle
width reaches zero. Button 3 reverses that transfer. Button 1 reverses the
upper transfer (subject to capacity). Four-high color-14 rail mark starts at
x=28, moving marker at x=8. Two button-2 clicks then five button-4 clicks
increase reward from 1 to 2. Independently, the interleaved sequence
4,4,4,2,4,2,4 also succeeds on a pristine clone. The leg uses this latter
policy with frame feedback: reduce marker separation, replenish empty middle
width when needed. Harness verification: levels=2 moves=10 replay_ok=True.

## Level 3: reservoir chain

Reproduced from fresh entry using `probe_level3.py`, `probe_transfers.py`, and
`probe_greedy.py`. Eight color-9 buttons form four horizontal pairs. Each
successful click transfers two pixels of stock height between adjacent tanks;
repeated clicks expose finite source stock and destination capacity. Initial
heights in two-pixel units are [2, 1, 2, 3, 14]. Moving three-pixel-wide marks
of colors 14, 15, and 11 match fixed two-pixel-wide marks at target heights
[8, 0, 0, 9, 5]. Prefix stock error supplies transport progress. A clone reached
reward 3 after 23 transfers; the candidate is in
`level3_transfer_candidate.json`. The new `align_reservoir_chain` leg derives
geometry and targets from the frame and tests progress on a clone before each
real move. `play_level_3` composes that leg only.

Validation: `python gkm_try.py` reported
`RESULT levels=3 moves=33 replay_ok=True err=None`.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_ujd0or1x/level3_transfer_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_bh2kb6jb/level4_float_route_candidate.json` and installed a thin replay player for the recovered suffix.

Level 5: verified horizontal, simultaneous figure exchange
- Reproduction: `python probe_level5.py` tests every observed button on a clone;
  `python probe_cross5.py` equalizes the first gate and tests gate/figure/mark clicks.
- Entry stock widths: (16,58,46,43); figure colors 11/14 occupy tanks 3/0.
  Gate right edges plus one are 40,52,37. Buttons transfer three columns.
- `python probe_plan5.py` reproduced that a gate exchanges both adjacent figures,
  and independently saved a winning 44-action suffix to
  level5_counterflow_candidate.json. Repeated rejected drainage probes established
  tank 2 cannot drain from 25 to 22, matching neighboring barrier geometry.
- New leg exchange_figures_through_locks uses joint stock/occupant BFS and verifies
  the complete route on a clone. Earlier float_through_locks supports only one
  figure and cannot express simultaneous exchanges, so its behavior is preserved.
- Final harness: RESULT levels=5 moves=98 replay_ok=True err=None.
  Supervisor-owned checkpoint.json was not modified.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_nameengg/level5_counterflow_candidate.json` and installed a thin replay player for the recovered suffix.
