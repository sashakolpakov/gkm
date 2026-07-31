# Unverified frontier brief: cn04 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The reproduced evidence identifies color `0` as the controlled 10-cell figure: actions 1–4 translate it by one 3-pixel lattice cell, action 5 rotates it, and action 6 is inert in the starting context. The naive reward search did not solve within depth 40, so I’m switching to a dense symbolic model: logical-cell silhouettes, separations, and contact/attachment changes among the four large figures.
- The first dense-progress signal is now concrete: after five downward moves and one rotation, two cells of the 32-cell top meter change from color 4 to the controlled color 0. This is a scored contact state, not a reward yet. I’m checking whether that score reflects attachment, overlap, or geometric edge matching before planning around it.
- The meter is persistent and advances when the moving silhouette passes specific peg/shape relationships; ordinary overlap with a colored figure does not advance it. I’ll now search for the next meter increment as a bounded subgoal, repeatedly, while checking the real level reward after every segment. This directly implements the required dense-progress strategy.
- The meter hypothesis failed a falsification check: it was simply the avatar overwriting the top barrier while being driven off-screen. I’m discarding that lineage. The reliable evidence remains the level-1 rewarded transition, so I’m reconstructing the exact final rotation geometrically and comparing its overlap/contact pattern with level 2.
- The reward search has gone beyond the small 3,000-state frontier and is still running within the 25,000-state cap. While it runs, the key geometric finding is that rotations can translate the figure when it engages nearby pegs, so level 2 is not reducible to a simple destination coordinate.
- The full-frame search exhausted 25,000 observed states without reward, so brute translation/rotation alone is insufficient. I’m now testing action 6 systematically across reachable contact and overlap contexts; it was inert only at the start and at the nearest C contact, which is not enough to classify it globally.
- Action 6 is now classified: it advances the turn counter but does not interact. The top strip is a time/move display (it advances on a wait as well as movement), which explains why the prior full-frame BFS exploded into time-stamped duplicates. I’ll collapse that display out of the observational key and rerun the reward search over actual board states.
- The level-1 rewarded sequence clarifies the role of the pegs: the first rotation covers the two pegs beside the target; the second covers the two below the avatar; only the third rewards. I’m mapping nearby placements to distinguish “visit all pegs” from an exact target-engagement condition, then I can formulate a meaningful level-2 dense objective.
- A stronger model now fits the rewarded level-1 geometry: the final pose seats the controlled piece against two side pegs beside the colored piece and on two support pegs. Level 2 likely requires an attachment chain. I’m testing that directly by searching for any state where a full colored component—not merely an occluded one—translates from its initial position.
- No single board pose rewards, even after collapsing search state to the avatar mask, so the level has path-dependent progress that is not rendered. I’m probing that safely with a systematic clone-only raster: visit every in-bounds translation in each of the four orientations while checking reward after every move. This tests the “visit required sockets/targets” model without guessing a hidden state key.
- The timer hard-stops the clone near 96 moves, so the systematic sweep correctly exposed the budget constraint. A much better dense measure now matches level 1 exactly: rotation 1 covers the two side pegs, rotation 2 covers the two support pegs, and the following rotation rewards. I’m reproducing the full peg-coverage trace, then I’ll search level 2 over `(avatar pose, visited-peg mask)` rather than raw frames.
- The peg hypothesis is useful but incomplete: a greedy route covered all 12 in 39 moves without reward, so visitation order matters. The layout supplies that order: the two pegs associated with the black start piece, four around color 14, four around color 11, and two around color 9—a staged chain consistent with level 1’s black-to-14 sequence. I’m now preserving group order instead of greedily taking the nearest peg.

## Preserved local probes

- `probe_level1_finish.py` (2014 bytes)
- `probe_level1_neighborhood.py` (993 bytes)
- `probe_level2.py` (1822 bytes)
- `probe_level2_attachment.py` (1291 bytes)
- `probe_level2_collect.py` (2478 bytes)
- `probe_level2_contact.py` (2953 bytes)
- `probe_level2_ordered.py` (2248 bytes)
- `probe_level2_progress.py` (1104 bytes)
- `probe_level2_reward.py` (741 bytes)
- `probe_level2_shapes.py` (1080 bytes)
- `probe_level2_sweep.py` (1889 bytes)
- `probe_level2_use.py` (1293 bytes)

Run or inspect the smallest relevant probe before writing another one.
