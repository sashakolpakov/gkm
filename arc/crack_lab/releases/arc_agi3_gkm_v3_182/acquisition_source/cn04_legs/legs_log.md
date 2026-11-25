# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Level 1: reproduced in probe.py and probe_contact.py from pristine entry.
Keys 1/2/3/4 translate the black piece and its two blue tips by 3 pixels
up/down/left/right. Key 5 rotates it one quarter turn; four turns restore
its original geometry. Yellow and its two blue tips remain fixed.
Three rotations, seven downward translations, and three rightward translations
place each movable blue tip adjacent to a fixed blue tip without reward.
One more rightward translation overlaps both pairs and increments levels to 1.
The reusable mate_blue_tips leg minimizes the number of actions among exact
blue-tip alignments, verifies candidates on clones, then commits the winner.

## Level 2: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_cn04_arc_agi3_n0_fresh_frontier_treudnhm/assembled_tips_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 6: reproduced in this workspace

`python gkm_try.py` reproduced the entry boundary: levels=5, moves=150.
`probe_level6.py` tests all five advertised key actions on separate clones.
Movement shifts the selected piece by three pixels; USE rotates rigid pieces.
Clicks on silhouette pixels reveal their body colors and tips while hiding
other pieces. `probe_extensions6.py` reproduces USE cycles for each silhouette:
two pieces extend, one over five steps and one over three, until their black
frontiers below the status row disappear. Several growth steps add no blue tips.

The new expose_marked_pieces leg selects and extends marked silhouettes,
reusing extend_marked_piece. The latter now checks body growth rather than
blue-tip growth, inferring background by area so earlier colors still work.
assemble_blue_tips now deduplicates partial pose assignments to avoid spending
its search bound on different insertion orders of the same geometry.

play_level_6 composes expose_marked_pieces and assemble_blue_tips. Both resumed
and fresh harness runs reach levels=6 in 191 moves with replay_ok=True.
The candidate capture script saves the fresh validated path separately as
level6_exposed_tips_candidate.json.
