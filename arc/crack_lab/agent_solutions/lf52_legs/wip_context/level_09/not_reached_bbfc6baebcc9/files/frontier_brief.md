# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The current level-9 suffix wastes 16 physical moves shuttling the remote bridge pair toward the carried survivor and then back again—the exact size of the missing 32 actions. I’m inspecting the revealed rail map for a cooperative alternative where the remote peg/bridge side finishes in place instead of performing that round trip.
- The world-model lower bound is sound for single-selection play, so I’m testing the one interaction pattern it omits: multi-selection. I’ll click two distinct pieces before a destination and compare every resulting physical state with the complete two-click outcome set; any novel grouped move would be the kind of compression the level needs.
- Another omission in the lower-bound model is selection persistence across scrolling. I’m testing “select a piece → move the carrier viewport several keys → click a new destination” against every ordinary move available at the final viewport. This could transfer an off-screen piece without the 16-move bridge shuttle.
- Selection survives visually across scrolling, but the far-to-left sweep produced no novel physical outcome across 1,784 source/step/destination combinations. The near-to-right sweep is still finishing; this is a direct falsification of “select off-screen, then teleport on a later click.”
- A different goal mechanism would most plausibly be disposal: moving either peg onto a wall/rail location not represented as a lattice cell. I’m reproducing the all-surface test at both revealed contexts with physical-state comparison, including fixed bridges and carriers as sources, and reporting only outcomes unreachable by ordinary cell moves.
- The exhaustive surface sweep is still running because it covers every even pixel, not just lattice centers. No novel outcome has appeared yet; both workers remain bounded to finite in-frame coordinates and will emit only a compact final result.
- The clean alternative is now clear: the old leg fully solves the 15-peg entry board before touching the remote frontier. A joint solution can keep temporary pegs as stepping stones, potentially eliminating the 16-move bridge round trip. I’m building a symbolic A* over the full initial world, with the admissible dense measure `2 × (pegs remaining − 1)`, then replaying any candidate on a clone.
- The joint replay exposed the real goal: “one peg” is insufficient. A clone reaches one peg in 8 actions with no reward; the successful states require that survivor to be loaded in the moving carrier. The existing world search stopped at the weaker condition. I’m correcting that goal and searching the intended 42-action suffix bound—the number that would make level 9 cost 70 and the fresh campaign exactly 600.
- There is one remaining integration gap: the 14-move opening proof used coordinate moves only. I found an 8-action route to one peg outside the carrier; if pre-frontier key actions can move the carrier to that survivor, the opening could become dramatically shorter. I’m searching that exact one-peg state with real clone transitions and a carrier-loaded goal.
- The first joint candidate failed because its geometry was observed after the transition but replayed before it. I’m correcting that alignment with one explicit transition action; if the four moves then genuinely leave one peg, there is still an 18-action window to reach the carrier and beat the 28-action opening.
- The clean lower bounds now conflict with the harness cap: level 9 needs 28 opening actions plus an exact 74-action frontier suffix, while the independently minimized prefix is 530. I’m doing one final resumed replay to record the production behavior and verify that the only failure is the 600-action cap—not an implementation error.
- Blocked by the harness’s move cap, not by the implementation. - Added `play_level_9` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n7_recover_clean_infrastructure_wip_xqqvp5o9/players.py:46), composing the verified level-9 leg. - Reproduced the level-9 lower bound: 28-action opening plus an exact 74-action suffix; no suffix exists below 74 over 120,566 modeled states. - Exhaustive valid-action probes found no multi-selection, off-lattice disposal, selection-scroll transfer, fixed-bridge movement, or direct carrier shortcut. - Final replay: `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap`. The current 544-action checkpoint leaves only 56 actions, while level 9 requires 102. Even the validated 530-action fresh prefix totals 632. `checkpoint.json` was not modified. Completion requires a supervisor-side cap increase to at least 646 for the current checkpoint, or a checkpoint/prefix of at most 498 actions.

## Preserved local probes

- `frontier_scaffold.json` (5718 bytes)
- `probe_action7_accounting.py` (355 bytes)
- `probe_action7_capture_undo.py` (1550 bytes)
- `probe_action7_contexts.py` (2877 bytes)
- `probe_action7_shifted_undo.py` (2427 bytes)
- `probe_alternate_controller_paths.py` (3553 bytes)
- `probe_bridge_solver_variant.py` (1843 bytes)
- `probe_capture_click_chaining.py` (967 bytes)
- `probe_capture_suffix_shortcut.py` (5986 bytes)
- `probe_carrier_reset_solver.py` (1703 bytes)
- `probe_carrier_weighted.py` (2842 bytes)
- `probe_coordinate_chaining.py` (2109 bytes)
- `probe_coordinate_controls.py` (2789 bytes)
- `probe_ddmin_level.py` (6023 bytes)
- `probe_delete_key_actions.py` (7589 bytes)
- `probe_fast_key_closure.py` (3456 bytes)
- `probe_fixed_move_dijkstra.py` (8445 bytes)
- `probe_goal_search_level4.py` (3763 bytes)
- `probe_inplace_l4_goal.py` (4881 bytes)
- `probe_inplace_l4_search.py` (5559 bytes)
- `probe_inplace_l7_final.py` (6636 bytes)
- `probe_key_neighborhood_events.py` (5458 bytes)
- `probe_key_reset_semantics.py` (2043 bytes)
- `probe_key_run_replay.py` (3782 bytes)
- `probe_key_run_shortcut.py` (6552 bytes)
- `probe_known_attach.py` (1111 bytes)
- `probe_l9.py` (9501 bytes)
- `probe_l9_alternatives.py` (20596 bytes)
- `probe_l9_ascii.py` (1431 bytes)
- `probe_l9_bridge_carried_openings.py` (5682 bytes)
- `probe_l9_bridge_shuttle.py` (5444 bytes)
- `probe_l9_carrier_click_moves.py` (2677 bytes)
- `probe_l9_carrier_pixels.py` (1881 bytes)
- `probe_l9_crossing.py` (6093 bytes)
- `probe_l9_empty_world.py` (1660 bytes)
- `probe_l9_far_clicks.py` (3405 bytes)
- `probe_l9_far_reset_unlock.py` (2654 bytes)
- `probe_l9_far_transfers.py` (2431 bytes)
- `probe_l9_fixed_moves.py` (1770 bytes)
- `probe_l9_frontier_reset.py` (1935 bytes)
- `probe_l9_frontier_undo_state.py` (1794 bytes)
- `probe_l9_hidden_agents.py` (3594 bytes)
- `probe_l9_hidden_turns.py` (2840 bytes)
- `probe_l9_joint_model.py` (7146 bytes)
- `probe_l9_key_tree.py` (2737 bytes)
- `probe_l9_late_turns.py` (2387 bytes)
- `probe_l9_layout_turns.py` (5570 bytes)
- `probe_l9_loaded_rail.py` (2191 bytes)
- `probe_l9_loads.py` (6953 bytes)
- `probe_l9_local_captures.py` (4817 bytes)
- `probe_l9_multiselect.py` (2989 bytes)
- `probe_l9_nonlocal_moves.py` (6670 bytes)
- `probe_l9_onepeg_finalize.py` (2775 bytes)
- `probe_l9_onepeg_openings.py` (3923 bytes)
- `probe_l9_opening_goals.py` (15173 bytes)
- `probe_l9_opening_mitm.py` (6468 bytes)
- `probe_l9_prereveal_keys.py` (1789 bytes)
- `probe_l9_reset_reveals.py` (1910 bytes)
- `probe_l9_reuse.py` (1463 bytes)
- `probe_l9_selected_keys.py` (2663 bytes)
- `probe_l9_selection_scroll.py` (3257 bytes)
- `probe_l9_shifted_load.py` (1705 bytes)
- `probe_l9_short_meet.py` (5886 bytes)
- `probe_l9_short_onepeg.py` (3861 bytes)
- `probe_l9_stage.py` (14974 bytes)
- `probe_l9_surface_moves.py` (2433 bytes)
- `probe_l9_variant_model.py` (5685 bytes)
- `probe_l9_world_model.py` (8713 bytes)
- `probe_leg_completion_index.py` (2263 bytes)
- `probe_level_controller_trace.py` (3069 bytes)
- `probe_level_coordinate_controls.py` (3911 bytes)
- `probe_level_event_closures.py` (6921 bytes)
- `probe_level_legal_moves.py` (2438 bytes)
- `probe_level_nonstandard_moves.py` (3683 bytes)
- `probe_long_coordinate_moves.py` (3939 bytes)
- `probe_mixed_reset_runs.py` (3831 bytes)
- `probe_multihop_contexts.py` (1439 bytes)
- `probe_optimize_level7.py` (7174 bytes)
- `probe_optimize_level9_path.py` (3882 bytes)
- `probe_post_reward_undo.py` (965 bytes)
- `probe_prefix_cycles.py` (1166 bytes)
- `probe_prefix_noops.py` (1220 bytes)
- `probe_prefix_structure.py` (2035 bytes)
- `probe_prior_entries.py` (1229 bytes)
- `probe_relaxed_key_run.py` (3178 bytes)
- `probe_repeated_reset_runs.py` (3127 bytes)
- `probe_reset_unit_shortcuts.py` (5251 bytes)
- `probe_route_beam.py` (8419 bytes)
- `probe_selected_directional_moves.py` (1004 bytes)
- `probe_shortcut_bridge_level.py` (9559 bytes)
- `probe_stage_click_effect.py` (3716 bytes)
- `probe_transition_undo.py` (1735 bytes)
- `probe_tune_bridge_solver.py` (2242 bytes)
- `probe_undo_goal_search.py` (7259 bytes)
- `probe_verify_integrated_legs.py` (1754 bytes)

Run or inspect the smallest relevant probe before writing another one.
