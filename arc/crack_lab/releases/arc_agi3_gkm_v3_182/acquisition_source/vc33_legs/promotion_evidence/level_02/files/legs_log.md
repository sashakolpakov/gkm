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
