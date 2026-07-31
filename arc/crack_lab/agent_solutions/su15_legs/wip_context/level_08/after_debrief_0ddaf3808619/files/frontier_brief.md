# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The improved star is born at `(41,43)`, but its unselected velocity is down-left, away from the right ring. I’m testing its bottom-right control pixel followed by a neutral turn to determine whether color-14 direction can be changed persistently; this is a two-step context check, not an assumption from the color-7 controls.
- The color-14 control check is decisive: clicking its pixels changes only its immediate position; the next autonomous velocity remains unchanged. So the star’s heading must be engineered at birth. I’m exhaustively rescoring the small two-turn merge frontier by a one-turn forecast—position plus whether its natural velocity points toward a right-hand ring.
- No complete two-turn merge has a favorable rightward forecast; the best valid star is born at `(47,31)` but moves away from the right rings. The depth-three frontier is now explicitly ranked by its next autonomous position, so any retained result will include heading compatibility, not just a visually attractive frame.
- The forecasted layer is computationally heavier than the raw position beam because every candidate is replayed one extra turn, but it remains bounded and has not produced a misleading intermediate result. I’ll stop it if this layer does not finish in the next window and use the verified two-turn lineage as a manual branch.
- The forecasted heading was indeed favorable—down-right—but its path crossed the staged square. That is recoverable: on the collision turn, move the square six cells right along the bottom lane so it stays just ahead of the star. I’m replaying that synchronized motion and checking whether both survive.
- The apparent favorable star forecast was only achieved by clicking near both the square and star; on the next turn they collide. I found the scoring bug: the forecast checked the star’s position but did not require the staged square to survive. I’m correcting that invariant and rerunning the same bounded three-turn frontier, with a smaller beam.
- The corrected beam retains only star states whose one-turn forecast still contains the intact color-8 square, color-12 square, full eight-cell color-14 star, and remaining color-7 cutter. The short frontier is stable; depth three is still evaluating those stricter candidates.
- The stricter layer is slower but bounded at 140 retained states, and it has not emitted a false “best” state. I’m letting this one layer finish because its output directly determines whether a safe star birth exists in three controls.
- The corrected lineage survives its forecast and gives the star a down-right velocity. At the forecast frame it is one turn from the bottom square, so I’ll use that very turn to send the square left while the star continues right—assigning them to opposite bottom rings and increasing separation each move.
- The synchronized move still couples the nearby star and square, so one-turn survival is insufficient. I’m returning to the earlier verified star birth at `(41,43)`, which leaves 27 cells of separation from the square. Its heading is down-left, but that distance lets me place the square in the bottom-left ring while the star travels independently.
- The well-spaced root now has two color-12 squares beside the left rings, a full color-14 star at `(49,35)`, and the remaining cutter at `(19,12)`. I’m launching a bounded placement beam from exactly that state, with a dense score for distinct-ring assignment and explicit collision clearance; states that lose any listed target are discarded.
- The placement beam is making verified progress: in two turns it centers the top square at `(19,8)`, keeps the bottom square at `(55,7)`, shifts the star right to `(53,33)`, and moves the cutter safely away. The dense assignment score improved from 26 to 24 with all target objects intact.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_beam8.py` (3293 bytes)
- `probe_beam_clean8.py` (4425 bytes)
- `probe_bfs23.py` (2174 bytes)
- `probe_body_controls8.py` (1396 bytes)
- `probe_clean8.py` (5740 bytes)
- `probe_cutter_goal.py` (2038 bytes)
- `probe_direction_persistence8.py` (1228 bytes)
- `probe_dots.py` (903 bytes)
- `probe_endpoint_bfs8.py` (5827 bytes)
- `probe_evolve8.py` (6640 bytes)
- `probe_exact_search8.py` (4032 bytes)
- `probe_finish8.py` (4381 bytes)
- `probe_frontier23.py` (3109 bytes)
- `probe_hold8.py` (1788 bytes)
- `probe_hold_persistence8.py` (1214 bytes)
- `probe_hold_search8.py` (5274 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level6_detail.py` (2088 bytes)
- `probe_level8.py` (5335 bytes)
- `probe_merge_star8.py` (6442 bytes)
- `probe_model8.py` (5969 bytes)
- `probe_patterns8.py` (913 bytes)
- `probe_place_targets8.py` (6061 bytes)
- `probe_root8_actions.py` (1824 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `probe_seed_finish8.py` (3077 bytes)
- `probe_stage8.py` (1963 bytes)
- `probe_state8.py` (1215 bytes)
- `probe_symbolic8.py` (1676 bytes)
- `probe_targets8.py` (5254 bytes)
- `probe_variants8.py` (5313 bytes)
- `probe_waits8.py` (2108 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
