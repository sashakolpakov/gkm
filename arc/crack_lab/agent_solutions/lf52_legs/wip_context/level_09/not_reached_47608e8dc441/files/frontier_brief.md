# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The lower carrier is a true dead-end rail; it cannot enter the shared network empty or loaded, so the direct-load branch is falsified. I’m now wiring the independently reward-verified 102-action level-9 leg and the already reproduced 530-action optimized prefix into the actual player/leg composition. This will establish the strongest verified implementation baseline before the final harness check.
- The wired fresh campaign is still running. The expected verification boundary is level 8 at 530 actions; if it reaches that point, the current 102-action level-9 leg will hit the cap at 600, confirming the remaining gap is exactly 32 actions rather than an implementation mismatch.
- Fresh replay is spending most of its time in the existing level-5 clone planner; no harness result has emitted yet. This is expected for a fresh run and does not consume extra real moves, but I’m keeping the single worker foreground and bounded by the harness wall-time.
- The level-9 remainder is now structurally accounted for: six net conveyor shifts are required to expose row 30, each shift forces bridge compensation, and the peg needs four horizontal plus two vertical jumps. The native 34-action phase meets those lower bounds exactly. The remaining leverage is level 7’s final carrier handoff. Its last carrier moves only six pixels but takes nine keys because of synchronized obstacles. I’m tracing those nine states to look for an alternative unload point or carrier choice.
- The level-7 handoff is also structurally constrained: the loaded peg has only one legal unload, and the vacated carrier is blocked by a fixed support until the nine-key synchronized route clears it. I’m shifting to a global search on level 4, whose shallow entry makes clone expansion practical and whose original planner was demonstrably nonoptimal (62 actions reduced to 50 by deletion alone).
- The level-4 global search is practical and has expanded 50 distinct macro states through action cost 19. It has not yet found a sub-50 reward path; I’m letting the bounded 300-state frontier run because this is the first search broad enough to change move order rather than merely delete actions.
- The first priority function over-favored states with three visible pegs and delayed necessary captures, reaching cost 33 before exploring cheaper two-peg states. I stopped it and am switching to action cost plus a capture lower bound, so the search behaves like A* toward a sub-50 solution.
- The corrected level-4 A* is now exploring in cost order: 125 states cover costs only through 18, with 193 unique parsed worlds. It has not found reward yet, but it is no longer trapped in high-cost reveal states.
- The global level-4 search is too broad to reach the known 50-action solution within a useful bound; through 175 expansions it had only covered costs to 21. I stopped it rather than spend the harness window proving a weak lower bound. I’m running the ordinary resumed harness now to record the exact implemented result and remaining cap gap.
- Component verification passes for the optimized level-4, level-6, and level-7 legs, and the ordinary harness confirms the wired level-9 route reaches the 600-action cap while remaining replay-valid. I’m preserving the exact 632-action full campaign candidate separately, as required for independent harness adoption; `checkpoint.json` remains untouched.
- The exact candidate is preserved and clone-verified at 632 actions with reward transition to level 9. I’m removing only the temporary search probes I added during this turn; the reusable legs, player composition, verifier, and `campaign_candidate_632.json` remain.
- Not completed: the harness still stops at the campaign cap: `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap` Implemented and verified: - Level-9 dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n2_continue_clean_wip_lalriwhd/players.py:46). - Clone-verified 102-action level-9 leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n2_continue_clean_wip_lalriwhd/legs.py:619). - Optimized level-4/6/7 prefix legs, reducing the level-8 boundary from 544 to 530 actions. - Preserved [campaign_candidate_632.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n2_continue_clean_wip_lalriwhd/campaign_candidate_632.json), independently clone-verified with transitions through level 9 at action 632. The remaining blocker is exactly 32 actions below the fixed 600-action cap. `checkpoint.json` was not modified.

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
- `probe_verify_best_campaign.py` (1677 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (27994 bytes)

Run or inspect the smallest relevant probe before writing another one.
