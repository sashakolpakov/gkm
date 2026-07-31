# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The compact level-4 leg does not generalize to level 5. I’m checking the analogous high-value reuse at the actual frontier: the level-8 grid-wrapped leg is 68 actions, which would put the optimized campaign at 599 if it solves level 9’s new wrapped configuration.
- The level-8 leg does not solve level 9. I’m broadening that same reuse check to the expensive prefix levels: if the 68-action grid leg solves level 7’s parallel layout, or the 92-action single-wrapped leg solves it, the campaign immediately fits. These are clone-only reward tests on pristine level entries.
- A new high-value configuration is available at pristine level-9 entry: key 4 moves the empty carrier before any peg clicks. The current abstract solver hard-codes the carrier at `(36,42)`; prepositioning it can change the collapse landing and wrapped reveal offset. I’m enumerating right-shifted carrier entries and solving each visible board abstractly before testing relay reuse.
- Straight right shifts to `(36,48/54/60)` are unsolvable: the local bridges cannot leave one peg in those carrier landings. I’m exploring the carrier’s small pre-click rail graph in all four directions after each turn, and will retain only positions where the abstract board actually has a one-peg carrier solution.
- Only the original carrier landing is solvable. The remaining timing variable is offscreen agency: entry actions 1–3 are visually inert, but may advance hidden wrapped pieces. I’m delaying the same verified 28-click collapse by 0–12 inert turns and grouping the resulting reveal geometries; a shifted bridge/peg phase could trade a few delays for a much shorter relay.
- Entry delays change only UI pixels; the physical reveal is identical. The frontier brief specifically flagged timing divergence after a capture, so I’m repeating the bounded delay at action 24—after the peg count drops—then finishing the last four clicks and grouping physical reveal components only.
- No bounded timing delay changes the physical reveal. I’ve exhausted the credible local shortcuts, so I’m running the required fresh replay with the composed 50-action level-4 leg, 92-action level-6 leg, and level-9 solver to establish the exact end-to-end result from current code before final cleanup.
- The fresh replay is still inside solver planning and has emitted no result yet. I’m letting the single end-to-end run continue to the harness boundary; it is the authoritative check for both action count and replay validity.
- No output after two minutes; this matches the known expensive level-5 clone planner rather than an action-cap failure. I’ll wait through the local run’s 180-second planning horizon, then report whether it reached a terminal harness result.
- The fresh run has exceeded 180 seconds without a harness result, so the planner wall time—not the real-action replay—is now the active blocker. I’m stopping it and will use the already reward-verified saved campaign path for action-count evidence.
- The current implementation is valid, but the verified campaign remains over the hard action budget: optimized prefix 531 plus the only reward-winning level-9 leg 102 equals 633. I’m cleaning exploratory probes, preserving only the solver change and original frontier artifacts, then I’ll run a final compile/default harness check and report the exact unresolved boundary.
- Not completed: the harness still reports: ```text RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap ``` Implemented [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont4_max180_20260728_zbnerjq6/players.py:2) to use the 50-action level-4 leg and compose the verified 102-action level-9 leg. Compilation passes. The reduced campaign remains 633 actions, 33 over budget. Verified observations are preserved in [frontier_brief.md](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_wip_cont4_max180_20260728_zbnerjq6/frontier_brief.md). `checkpoint.json` was untouched.

## Preserved local probes

- `campaign_candidate_633.json` (11978 bytes)
- `level4_ddmin_50.json` (1085 bytes)
- `level6_ddmin_92.json` (2021 bytes)
- `level9_candidate_102.json` (1355 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `probe_ddmin_level.py` (2912 bytes)
- `probe_level9_abstract_entry.py` (3812 bytes)
- `probe_level9_reproduce.py` (4363 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (4431 bytes)

Run or inspect the smallest relevant probe before writing another one.
