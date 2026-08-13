# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The neutral-turn orbit did not finish within a useful clone budget. I’m tightening the test: search directly for the known *pre-macro* frame, and validate the coordinate macro only when that frame is reached. This removes two clone steps from every rejected state while preserving an exact drop-in criterion.
- Full-frame keying is distinguishing decorative rail/background translations that do not affect a bridge macro. For levels 4–5 I can safely key the search by the documented puzzle geometry—slots, pegs, carriers, fixed bridges, borders, and selection—then replay any found path against the full suffix. I’m applying that compact key only to the probe.
- Exact neutral-turn BFS is too clone-heavy to finish at the long level-5 transfer, and the shorter level-5 groups showed no gain. A faster reward-backed test is available for level 7: replace contiguous key runs with one neutral turn, replay the entire known solution on a clone, and greedily retain only replacements that still earn the level reward. This directly measures campaign savings without requiring state equality.
- The level-7 reward-backed replacement sweep is still within its 300-test cap and has found no successful multi-key-to-neutral replacement so far. I’m letting it finish; unlike frame matching, every accepted result here would be a directly replayable campaign saving.
- No replacement has succeeded yet. The sweep is replay-cost bound rather than state-space bound; I’ll allow one more minute, then stop if it still has no positive result and move to targeted single-key neutral substitutions, which are cheaper and can reveal whether `7` matters at all in level 7.
- The neutral-turn drop-in sweep found no savings. I’m returning to the strongest unresolved macro branch from the complete level-7 orbit: unloading the first peg at lower column 42 costs one extra key, but leaves the carrier in the same column as the upper bridge. If that cuts the following eight-key return trip, it can produce a genuine net gain.
- The far unload is falsified—it makes the bridge return one key worse. A more promising scheduling opportunity is now explicit: after the peg reaches the lower row, it has three useful zero-key relay jumps available while the empty carrier must travel back to the upper bridge. Because every click advances the world, I’m interleaving the first peg jump and remeasuring the carrier alignment instead of treating those actions as separate phases.
- The neutral-turn orbit found the missing cooperative state: after loading the peg, `7,1,4,4,4,4,4,4` exposes a second carrier and legally loads the bridge without unloading the peg. This replaces the admitted unload–return–reload sequence. I’m preserving that dual-loaded prefix and searching for the cheapest synchronized unloads.
- The dual-loaded orbit is larger because both carriers can move independently while `7` preserves momentum. The current search remains capped at 900 physical states; I’m waiting for its validated unload options rather than guessing coordinates, since one wrong cargo assignment would invalidate the branch.
- The 900-state brute validator is too slow because it retests every destination at every state. I’m stopping it and narrowing the branch to observable cargo coordinates plus the two known lower-row unload endpoints; this keeps the same clone evidence with far fewer probes.
- The first neutral path did not create a dual-loaded state: while it exposed the bridge carrier, the peg returned to its original upper slot. I’m refining the orbit key to distinguish identical bridge moves by their resulting cargo state, then checking whether *any* short path loads the bridge while the peg remains transported.
- The refined orbit exhausts at 48 states: `7` can switch which cargo assignment is active, but no state holds both the peg and bridge in carriers. That branch is now falsified. I’ll apply the same complete neutral-turn option audit to level 5’s fixed-bridge carriers, where alternate carrier assignment could still shorten its 57 key turns.

## Preserved local probes

- `campaign_candidate_632.json` (11973 bytes)
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
- `level7_bridge_first_probe.json` (228 bytes)
- `level7_bridge_then_peg_probe.json` (281 bytes)
- `level7_dual_loaded_probe.json` (136 bytes)
- `level7_greedy_macro_candidate.json` (1915 bytes)
- `level7_interleaved_peg_probe.json` (166 bytes)
- `level7_key_shortcuts_candidate.json` (1920 bytes)
- `level7_macro_ddmin_candidate.json` (1915 bytes)
- `level7_neutral_replacement_candidate.json` (1915 bytes)
- `level7_peg_far_unload_probe.json` (141 bytes)
- `level8_greedy_macro_candidate.json` (1435 bytes)
- `level8_key_shortcuts_candidate.json` (1435 bytes)
- `level9_candidate_102.json` (1355 bytes)
- `level9_entry_variant_candidate.json` (2731 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `level9_macro_ddmin_candidate.json` (2731 bytes)
- `level9_walled_mixed_solution.json` (3726 bytes)
- `probe_all_coordinate_moves.py` (3628 bytes)
- `probe_bridge_key_geometry.py` (1448 bytes)
- `probe_bridge_options.py` (2376 bytes)
- `probe_bridge_trace.py` (2162 bytes)
- `probe_carrier_click.py` (2133 bytes)
- `probe_composed_reuse.py` (1879 bytes)
- `probe_cycle_remove.py` (2189 bytes)
- `probe_ddmin_level.py` (2912 bytes)
- `probe_ddmin_level9_macros.py` (4016 bytes)
- `probe_greedy_macro_deletions.py` (2824 bytes)
- `probe_inert_action_deletions.py` (2824 bytes)
- `probe_l9.py` (3601 bytes)
- `probe_l9_bfs.py` (3577 bytes)
- `probe_l9_candidate.py` (9207 bytes)
- `probe_l9_carrier_border.py` (1655 bytes)
- `probe_l9_clipped.py` (1503 bytes)
- `probe_l9_moves.py` (2017 bytes)
- `probe_l9_search.py` (2071 bytes)
- `probe_l9_shortcuts.py` (1629 bytes)
- `probe_l9_special.py` (2051 bytes)
- `probe_l9_stage1_abstract.py` (2560 bytes)
- `probe_l9_symbolic_global.py` (4615 bytes)
- `probe_l9_wrap_shortcut.py` (1900 bytes)
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
- `probe_level9_saved_path.py` (909 bytes)
- `probe_level9_shortcut_search.py` (5342 bytes)
- `probe_level9_shortest_suffix.py` (8059 bytes)
- `probe_level9_splice_suffixes.py` (2516 bytes)
- `probe_level9_stages.py` (2238 bytes)
- `probe_level9_undo_edges.py` (1992 bytes)
- `probe_minimize_path.py` (3332 bytes)
- `probe_movable_options2.py` (4322 bytes)
- `probe_movable_trace.py` (1970 bytes)
- `probe_neutral_run_replacements.py` (3286 bytes)
- `probe_optimal_bridge_level.py` (3137 bytes)
- `probe_optimize_level7.py` (4965 bytes)
- `probe_prefix_counts.py` (1105 bytes)
- `probe_prefix_suffix_matches.py` (6854 bytes)
- `probe_preselected_carrier.py` (1665 bytes)
- `probe_selected_actions.py` (2149 bytes)
- `probe_shorter_key_segments.py` (3616 bytes)
- `probe_shorter_segment.py` (2986 bytes)
- `probe_suffix_splices.py` (4611 bytes)
- `probe_undo_phase.py` (2074 bytes)
- `probe_undo_replacements.py` (3106 bytes)
- `probe_verify_best_campaign.py` (1677 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (38422 bytes)

Run or inspect the smallest relevant probe before writing another one.
