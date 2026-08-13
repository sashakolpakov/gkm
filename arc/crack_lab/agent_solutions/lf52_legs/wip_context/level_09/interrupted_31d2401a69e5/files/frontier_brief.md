# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- Longer level-6 opening assignments cannot load both cargo carriers, so the canonical 11-macro assignment is unique. I’m closing one unresolved branch precisely: the nine-key relay entry plus its four-key bridge correction. If it converges to the canonical post-macro state, it is formally dominated by one action; if not, its later carrier phase remains worth testing.
- One assumption in the level-9 lower bound needs widening: the first board may not have to be consolidated before transport. If either original peg can enter the carrier while the other remains, the extra peg could serve as a remote support and the carrier could return for the final capture. I’m computing the exact minimum to any loaded-carrier state, not just the one-peg state.
- Level 9 cannot transport an unconsolidated peg or a bridge: exact BFS finds the first carrier load only after the same 14 macros, and a bridge can never enter that carrier. The remaining unproven route is level 8 after its opening. I’m extending the option enumerator to its color-9 bridges and checking the 12-key wrapped junction for alternate transfers.
- Level 8 exposes a real branch: its first wrapped bridge move is legal after 9 keys, not the canonical 12. The final three left moves only establish a particular carrier state. I’m following the shorter state through the remaining nine macros to see whether it converges with a net saving.
- The level-8 branch’s three-key saving is exactly spent later: after the same three bridge macros, the peg requires three left moves before its downward jump. I’m verifying state convergence; if it matches canonical, this branch is tied and provides no prefix reduction.
- A second level-8 carrier-state tradeoff is larger: after macro 16, the next bridge jump is already legal with zero keys; the canonical route spends four rights before making it. I’m following the immediate jump. If its later carrier correction costs fewer than four, this becomes a genuine prefix saving.
- The zero-key level-8 bridge branch also ties: after the three bridge jumps, the peg needs exactly the four omitted rights before its transfer. Carrier motion commutes with those bridge moves. I’m checking the final capture junction for any non-commuting alternative, then level 8 will be fully accounted.
- The 30-action gap matches the bridge shuttle exactly, suggesting the remote pair should wrap rather than walk linearly. At the right edge, a bridge can jump from world 106 to 118; after one more carrier step, the other can attempt the symmetric landing beyond 118. I’m probing that fully in-frame wrap sequence now.
- The apparent bridge wrap stops at world column 118; the beyond-edge landing is invalid. One remaining affordance could invalidate the relay model entirely: the carrier’s color-11 border may be a separate selectable object even though clicking its peg-filled interior only selects the peg. I’m testing valid border pixels and their highlighted landings.
- The carrier border is not a separate affordance; every border pixel selects the loaded peg and exposes only its ordinary leftward jump. The documented mechanics now leave a verified 30-action shortfall. I’m preserving the fully replayed 102-action level-9 path in a descriptive JSON file, as requested, while keeping `checkpoint.json` untouched.
- The preserved JSON has exactly 102 actions and matches the composed leg structurally. I’m replaying that file itself from the validated level-9 entry clone so the saved artifact—not merely the in-code route—is independently verified.
- Implemented and clone-verified level 9, but the campaign cap still prevents `RESULT levels>=9`. - [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n5_max_coherence_reset_1fz6_48s/legs.py:622) contains the reusable level-9 leg. - [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n5_max_coherence_reset_1fz6_48s/players.py:47) composes it. - [level9_walled_mixed_solution.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n5_max_coherence_reset_1fz6_48s/level9_walled_mixed_solution.json:1) preserves the independently replayed 102-action path; it reaches level 9 on a clone. - Prefix optimizations save 16 actions, reducing the fresh level-9 boundary from 544 to 528. - Current harness result: `RESULT levels=8 moves=600 replay_ok=True`. - The reproduced mechanics require 28 actions for the first board and 74 for the relay, leaving a verified 30-action shortfall under the cap. `checkpoint.json` was not modified.

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
- `level7_greedy_macro_candidate.json` (1915 bytes)
- `level7_key_shortcuts_candidate.json` (1920 bytes)
- `level7_macro_ddmin_candidate.json` (1915 bytes)
- `level8_greedy_macro_candidate.json` (1435 bytes)
- `level8_key_shortcuts_candidate.json` (1435 bytes)
- `level9_candidate_102.json` (1355 bytes)
- `level9_entry_variant_candidate.json` (2731 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `level9_macro_ddmin_candidate.json` (2731 bytes)
- `level9_walled_mixed_solution.json` (3726 bytes)
- `probe_bridge_key_geometry.py` (1448 bytes)
- `probe_bridge_options.py` (2373 bytes)
- `probe_bridge_trace.py` (1978 bytes)
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
- `probe_movable_trace.py` (1970 bytes)
- `probe_optimal_bridge_level.py` (3137 bytes)
- `probe_optimize_level7.py` (4965 bytes)
- `probe_prefix_counts.py` (1105 bytes)
- `probe_prefix_suffix_matches.py` (6854 bytes)
- `probe_preselected_carrier.py` (1665 bytes)
- `probe_selected_actions.py` (2149 bytes)
- `probe_shorter_key_segments.py` (3011 bytes)
- `probe_shorter_segment.py` (2986 bytes)
- `probe_suffix_splices.py` (4611 bytes)
- `probe_undo_replacements.py` (3106 bytes)
- `probe_verify_best_campaign.py` (1677 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (33232 bytes)

Run or inspect the smallest relevant probe before writing another one.
