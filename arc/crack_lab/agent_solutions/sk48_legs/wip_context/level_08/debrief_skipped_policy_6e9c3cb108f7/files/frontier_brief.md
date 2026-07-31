# Unverified frontier brief: sk48 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 7 at 584 actions.
- Remaining real-action budget under the harness cap: 16 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The reward is now independently confirmed at level-8 move 134. Because this route deliberately over-stages objects, I’m minimizing its primitive action path directly on pristine clones: first delete large contiguous chunks, then single actions, accepting a deletion only when the real level reward still reaches 8. This preserves mechanics while exposing the genuinely necessary core.
- Reward-preserving deletion compressed the route from 134 to 85 moves in one pass, including removal of the entire initial `8→12` staging block. Since deletions can make earlier actions newly redundant, I’m running minimization to a fixed point before deciding how much prefix optimization is still required.
- The fixed-point pass has reached 63 moves and is still finding reward-preserving deletions. This is now in the range anticipated by the frontier, and it cuts the remaining prefix requirement to roughly 47 moves instead of 69.
- The level-8 path is fixed-point minimal under deletion at 63 moves and wins on a pristine clone. The fresh composed prefix must therefore fall from 586 to at most 537. I’m minimizing levels 5–7 against exact post-level frames—not just reward—so every accepted deletion preserves the next level’s carry-over state and existing downstream legs remain valid.
- Exact-frame minimization is much more productive than expected: level 5 has fallen from 189 to 91 moves while preserving its pixel-identical level-6 entry. That alone creates enough campaign room for the 63-move level-8 leg; I’m allowing the fixed-point pass to finish so the final implementation has margin rather than sitting at the cap.
- Level 6 is also shrinking while preserving its exact level-7 entry, though less dramatically. The campaign already has over 30 moves of safety margin from level 5 alone, so these additional deletions are robustness margin rather than a dependency.
- The second fixed-point pass on level 6 found another coupled deletion, confirming why simple “remove every unchanged turn” was unsafe. Every retained shortening here is validated by exact exit-frame equality.
- Level 6 is now at 127 moves (35 saved) and the minimizer is checking whether that is a fixed point. Combined with level 5, the prefix reduction is already 133 moves—ample room for the verified level-8 route.
- I have exact-frame-minimized paths for levels 5–7 (91, 127, and 50 moves) plus the 63-move level-8 path. I’m packaging each as a named leg behind one general action-path executor, updating players to compose only those legs, and then I’ll run the required fresh replay from level 1.
- Fresh replay now succeeds cleanly: `levels=8`, `moves=506`, `replay_ok=True`. The supervisor checkpoint remains untouched, so I’m exporting that validated 506-action fresh path to a separate descriptive candidate JSON for independent adoption, exactly as requested.
- The saved candidate parses to 506 actions and independently validates for 8 levels. I’m doing one final fresh boundary measurement to confirm each composed player advances exactly one level and that level 8 completes at the expected total, without touching `checkpoint.json`.
- Implemented and verified. - Fresh replay: `RESULT levels=8 moves=506 replay_ok=True err=None` - Level boundaries: `26, 72, 35, 42, 91, 127, 50, 63` - Added `play_level_8` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_sk48_sk48_l8_max180_wip_retry_20260728/players.py:51) - Added the shelf-handoff leg and compact prefix legs in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_sk48_sk48_l8_max180_wip_retry_20260728/legs.py:374) - Saved the independently validated 506-action path in [fresh_replay_level8_candidate.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_sk48_sk48_l8_max180_wip_retry_20260728/fresh_replay_level8_candidate.json:1) `checkpoint.json` was untouched. Its existing 584-move prefix remains stale until the supervisor adopts the saved candidate.

## Preserved local probes

- `fresh_replay_level8_candidate.json` (1274 bytes)
- `measure_levels.py` (3655 bytes)
- `minimize_level8_path.py` (2991 bytes)
- `minimize_prefix_paths.py` (2939 bytes)
- `probe_level8.py` (3775 bytes)
- `probe_level8_candidates.py` (5357 bytes)
- `probe_level8_mechanics.py` (8315 bytes)
- `search_level8.py` (4911 bytes)
- `test_prefix_deletions.py` (2673 bytes)
- `trace_level8.py` (3148 bytes)

Run or inspect the smallest relevant probe before writing another one.
