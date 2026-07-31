# Verified frontier update: wa30 level 9 complete

- Official harness result: `levels=9 moves=597 replay_ok=True err=None`.
- The validated checkpoint now reaches level 8 in 536 moves.
- `play_level_9` composes `stage_three_deliveries_dismiss_thief_and_finish`.
- Fresh replay also completes level 9 in 597 moves.
- The older 587-move checkpoint budget and all proposer notes below are
  superseded; retain them only as historical probe indices.

# Superseded unverified brief

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 587 actions.
- Remaining real-action budget under the harness cap: 13 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- I’ll stay inside this workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then reproduce any inherited observations from `frontier_brief.md` before relying on them.
- The inherited brief is internally mixed: its level-9 budget claim is useful only as a lead, while its detailed mechanics describe an earlier level. I’ll treat both as unverified and run the harness now to establish the real entry state and action budget.
- The pristine entry is confirmed indirectly by the harness: all eight existing players complete, and dispatch stops before any level-9 action. I’m adding one compact probe that replays those players, then reports only 4×4 symbolic cells, object components, and one-step clone deltas—no raw-frame dump.
- The first probe exposed an important harness detail: calling `solve.py` alone starts from level 1 and stalls at level 7, while `gkm_try.py` first replays its validated workspace checkpoint. I’ll mirror that documented resume path without printing or mining it, so the probe observes the same pristine level-9 state as the real harness.
- The level-9 entry is now reproduced: the avatar is the `(0,14)` cell at macro `(8,8)`, all four movement actions move it, and `USE` is inert in open space. Two autonomous objects move every turn—color 12 westward near row 7 and color 15 northward on the far right—so any solution must account for both. I’m comparing this layout only against the indexed courier legs most likely to match.
- Pure waiting establishes the first dense measure: the west courier repeatedly converts a loose `(4,9)` block into a carried `(5,9)` block and deposits it into the 3×3 outlined pad. Four such pickup/drop cycles occur in the first 40 turns. The color-15 object follows the lower maze and is behaving like a deadline runner, not cargo. I’m extending this one pristine clone only far enough to see whether autonomous play can ever complete, logging pickup/drop events instead of every pixel change.

## Preserved local probes

- `probe3_last_drop.py` (2169 bytes)
- `probe4_last_drop.py` (2161 bytes)
- `probe5_candidates.py` (1216 bytes)
- `probe5_structure.py` (1413 bytes)
- `probe5_tail_bfs.py` (1181 bytes)
- `probe5_tail_variants.py` (1550 bytes)
- `probe8_balance_lower.py` (3453 bytes)
- `probe8_candidates.py` (4200 bytes)
- `probe8_combo.py` (3142 bytes)
- `probe8_combo_finish.py` (1783 bytes)
- `probe8_current_tail.py` (1896 bytes)
- `probe8_drop_variants.py` (1485 bytes)
- `probe8_entry.py` (1367 bytes)
- `probe8_fast_finish.py` (1475 bytes)
- `probe8_lower_manual.py` (1541 bytes)
- `probe8_minimize.py` (2312 bytes)
- `probe8_optimized_tail.py` (1506 bytes)
- `probe8_reposition.py` (2565 bytes)
- `probe8_reverse_finish.py` (1688 bytes)
- `probe8_reverse_idle.py` (1498 bytes)
- `probe8_reverse_search.py` (2709 bytes)
- `probe8_reverse_stage.py` (1789 bytes)
- `probe8_tail_bfs.py` (871 bytes)
- `probe8_tail_mutations.py` (2251 bytes)
- `probe8_third_minimize.py` (750 bytes)
- `probe8_trace.py` (2423 bytes)
- `probe9.py` (13826 bytes)
- `probe9_actual_candidates.py` (4474 bytes)
- `probe9_actual_mutations.py` (2597 bytes)
- `probe9_actual_ports.py` (8832 bytes)
- `probe9_alt.py` (2466 bytes)
- `probe9_beam52.py` (3505 bytes)
- `probe9_below_pick.py` (2167 bytes)
- `probe9_best_mutations.py` (3689 bytes)
- `probe9_best_short_verify.py` (1522 bytes)
- `probe9_best_trace.py` (1409 bytes)
- `probe9_candidate.py` (3993 bytes)
- `probe9_clean_candidates.py` (5001 bytes)
- `probe9_clean_entry.py` (1689 bytes)
- `probe9_combined_variants.py` (1477 bytes)
- `probe9_context.py` (1760 bytes)
- `probe9_courier.py` (1563 bytes)
- `probe9_current.py` (19683 bytes)
- `probe9_delay_dismiss.py` (1711 bytes)
- `probe9_delete_one.py` (4252 bytes)
- `probe9_early_dismiss.py` (1691 bytes)
- `probe9_entry.py` (2521 bytes)
- `probe9_exact_finish_search.py` (2766 bytes)
- `probe9_fast_suffix.py` (2641 bytes)
- `probe9_full_short.py` (1651 bytes)
- `probe9_handoff.py` (1261 bytes)
- `probe9_idle_initial.py` (803 bytes)
- `probe9_minimize_win.py` (774 bytes)
- `probe9_phase_dismiss_search.py` (3116 bytes)
- `probe9_picksearch.py` (3443 bytes)
- `probe9_picksearch_finish.py` (2098 bytes)
- `probe9_pickstate.py` (1320 bytes)
- `probe9_position_beam.py` (3824 bytes)
- `probe9_position_nine.py` (2668 bytes)
- `probe9_position_orders.py` (4810 bytes)
- `probe9_position_ten.py` (2629 bytes)
- `probe9_prefix_shortcuts.py` (3955 bytes)
- `probe9_reroute.py` (2385 bytes)
- `probe9_reverse.py` (2306 bytes)
- `probe9_reverse_combined.py` (3521 bytes)
- `probe9_right_depot.py` (4919 bytes)
- `probe9_route61_minimize.py` (793 bytes)
- `probe9_route61_splice.py` (836 bytes)
- `probe9_route61_trace.py` (1373 bytes)
- `probe9_search.py` (13865 bytes)
- `probe9_short_finish_search.py` (2557 bytes)
- `probe9_short_idle.py` (1246 bytes)
- `probe9_short_pick.py` (2248 bytes)
- `probe9_short_place_search.py` (2668 bytes)
- `probe9_short_position_orders.py` (4072 bytes)
- `probe9_short_stage.py` (1658 bytes)
- `probe9_short_tail_search.py` (3249 bytes)
- `probe9_stage_endings.py` (2062 bytes)
- `probe9_stage_finish_beam.py` (934 bytes)
- `probe9_stageports.py` (2512 bytes)
- `probe9_structure_compact.py` (1760 bytes)
- `probe9_suffix.py` (2795 bytes)
- `probe9_tail_beam_wide.py` (4004 bytes)
- `probe9_thieftrace.py` (1222 bytes)
- `probe9_three_stage.py` (1392 bytes)
- `probe9_top_second.py` (1428 bytes)
- `probe9_turn55_beam.py` (1043 bytes)
- `probe9_two_stage_beam.py` (3205 bytes)
- `probe9_two_staged_trace.py` (1635 bytes)
- `probe9_upper_local_actual.py` (2009 bytes)
- `probe9_upper_pick.py` (2236 bytes)
- `probe9_variants.py` (3719 bytes)
- `probe9_verify.py` (7540 bytes)
- `probe_cumulative.py` (826 bytes)
- `probe_current_solver.py` (519 bytes)
- `probe_fresh_segment_minimize.py` (1036 bytes)
- `probe_interactions.py` (1663 bytes)
- `probe_level9_entry.py` (4551 bytes)
- `probe_minimize_segments.py` (3388 bytes)
- `probe_pair_minimize.py` (4700 bytes)
- `probe_placement.py` (1117 bytes)
- `probe_solution.py` (2372 bytes)
- `probe_solver_timing.py` (980 bytes)
- `probe_tiles.py` (1126 bytes)
- `verify_planner.py` (642 bytes)
- `wip_glue_notes.md` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
