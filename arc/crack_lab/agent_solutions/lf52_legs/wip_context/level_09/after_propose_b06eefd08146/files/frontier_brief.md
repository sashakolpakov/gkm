# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The undo orbit is now mapped: it can shift the loaded carrier among columns 34, 46, and 58, but it never changes rows or rewards by itself. I’m switching to a segmented search from the verified 12-action conveyor checkpoint; this avoids re-exploring the expensive entry orbit and directly tests whether the remaining campaign can finish within 30 actions (the fresh-replay budget).
- The segmented checkpoint search has covered all branches through cost 14 without reward. It remains bounded to a 30-action suffix, which would make the optimized fresh campaign fit exactly under 600 if found.
- I’m testing a concrete routing compression now: at the rightmost conveyor stop, the candidate moves only one bridge before reversing, then spends seven more jumps rebuilding the same chain. I’m staging progressively more of that chain before the reversal to see whether the remote peg re-enters with its vertical route already assembled.
- A promising branch has now reached two visible pegs only 12 pixels apart at suffix cost 21 from the segmented checkpoint. That leaves nine actions for support alignment and capture, so this branch can still fit the 600-action fresh campaign.
- The promising path is now reconstructed exactly. It moves the remote peg from `(12,22)` to `(24,22)`, leaving it directly between the loaded peg at `(36,22)` and the vacated landing at `(12,22)`. I’m testing the resulting long vertical capture; if accepted, level 9 costs 63 actions and the verified 530-action prefix yields a 593-action campaign.
- The long capture itself is illegal because this phase only accepts 12-pixel jumps. The dense state is still useful: it has three legal moves and leaves nine actions under the fresh campaign cap. I’m exhaustively searching that small final neighborhood now.
- The nine-action neighborhood is conclusively exhausted with no reward. Extending it to 20 actions has found another distance-12 arrangement at cost 14, but still no legal capture; this suggests the carrier/support landing, not raw peg distance, is the decisive remaining constraint.
- The distance-12 branch exposed the missing construction. Reversing one additional conveyor step places the remote peg at column 28; the same bridge can then act three times in succession: support the peg down to row 24, jump over that peg to row 30, and support it down to row 36. The loaded peg can then capture horizontally onto its carrier. This predicts a 68-action level-9 leg and a 598-action optimized campaign; I’m clone-verifying it now.
- The exact eight-action remainder is now being searched with undo-aware state keys from the column-28 staging state. No reward has appeared yet; undo can alter bridge routing there, but it has not exposed the required row-30 landing.
- The short-route hypothesis is falsified: the wrapped board does not expose row 30 at that offset, and the undo-aware eight-action remainder did not find a reward state before the bounded cap. I’m preserving the fully clone-verified 102-action multi-bridge leg in the library and wiring level 9 to it, while keeping the campaign-cap limitation explicit in verification.
- The remaining compression attempts are falsified, and the real harness still stops at 600 moves. I’m cleaning the exploratory scaffolding now while retaining only the clone-verified reusable level-9 leg and its player dispatch; I’ll report the exact remaining blocker rather than claim a campaign success that the harness does not show.
- Not completed: final harness result is: `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap` I added a clone-verified 102-action level-9 leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n1_continue_clean_wip_i_2ixay9/legs.py:593) and dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n1_continue_clean_wip_i_2ixay9/players.py:45). It reaches level 9 on a clone, but the 544-action checkpoint leaves only 56 real actions, so the campaign still exceeds the cap.

## Preserved local probes

- `campaign_candidate_633.json` (11978 bytes)
- `frontier_scaffold.json` (5718 bytes)
- `level4_ddmin_50.json` (1085 bytes)
- `level4_greedy_macro_candidate.json` (1085 bytes)
- `level5_ddmin_89.json` (1274 bytes)
- `level5_key_shortcuts_candidate.json` (1274 bytes)
- `level5_macro_ddmin_candidate.json` (1274 bytes)
- `level6_ddmin_92.json` (2021 bytes)
- `level6_greedy_macro_candidate.json` (2021 bytes)
- `level6_key_shortcuts_candidate.json` (2026 bytes)
- `level7_greedy_macro_candidate.json` (1915 bytes)
- `level7_key_shortcuts_candidate.json` (1920 bytes)
- `level7_macro_ddmin_candidate.json` (1915 bytes)
- `level8_greedy_macro_candidate.json` (1435 bytes)
- `level8_key_shortcuts_candidate.json` (1435 bytes)
- `level9_candidate_102.json` (1355 bytes)
- `level9_entry_variant_candidate.json` (2731 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `level9_macro_ddmin_candidate.json` (2731 bytes)
- `probe_carrier_click.py` (2133 bytes)
- `probe_composed_reuse.py` (1879 bytes)
- `probe_ddmin_level.py` (2912 bytes)
- `probe_ddmin_level9_macros.py` (4016 bytes)
- `probe_greedy_macro_deletions.py` (2824 bytes)
- `probe_inert_action_deletions.py` (2824 bytes)
- `probe_leg_reuse.py` (3154 bytes)
- `probe_level9_abstract_entry.py` (3812 bytes)
- `probe_level9_carrier_opportunities.py` (5545 bytes)
- `probe_level9_context_actions.py` (2272 bytes)
- `probe_level9_context_search.py` (1473 bytes)
- `probe_level9_direct_branch.py` (1806 bytes)
- `probe_level9_direct_finish.py` (3235 bytes)
- `probe_level9_entry_variants.py` (6196 bytes)
- `probe_level9_entry_worlds.py` (5128 bytes)
- `probe_level9_key_orbit.py` (1128 bytes)
- `probe_level9_labeled_entry_worlds.py` (7235 bytes)
- `probe_level9_macro_symbols.py` (3251 bytes)
- `probe_level9_reproduce.py` (4363 bytes)
- `probe_level9_shortcut_search.py` (5342 bytes)
- `probe_level9_shortest_suffix.py` (8059 bytes)
- `probe_level9_splice_suffixes.py` (2516 bytes)
- `probe_level9_stages.py` (2238 bytes)
- `probe_level9_undo_edges.py` (1992 bytes)
- `probe_prefix_suffix_matches.py` (6854 bytes)
- `probe_preselected_carrier.py` (1665 bytes)
- `probe_selected_actions.py` (2149 bytes)
- `probe_suffix_splices.py` (4611 bytes)
- `probe_verify_best_campaign.py` (1518 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (22756 bytes)

Run or inspect the smallest relevant probe before writing another one.
