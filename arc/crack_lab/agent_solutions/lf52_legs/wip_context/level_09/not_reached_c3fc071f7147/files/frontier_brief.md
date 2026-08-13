# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The detour beam is producing genuinely distinct synchronized states: two after stage 9 and six after stage 11, with only one or two extra actions. This is the first search in this pass that preserves alternate hidden-carrier configurations, so I’m letting it propagate through the wrap.
- Stage 13 is the expensive 13-key carrier delivery: nine incoming states times roughly 100 one-edit/insertion candidates. The probe remains bounded by the beam and worker allowance; no unbounded sequence enumeration is occurring.
- The width-32 run exhausted the worker allowance during stage 13 before producing a result. I’m narrowing to the four cheapest distinct states; that preserves the meaningful detours found so far while keeping the long handoff within the bounded action budget.
- The narrowed detour beam reached stage 20 with eight valid synchronized states: canonical cost 123 and alternatives at 124. The worker allowance ended before the final two runs. I’m reducing to the canonical state plus the cheapest detour; this should fit end-to-end and directly answer whether one extra earlier key removes a longer tail.
- Both cheapest states have reached the final delivery milestone at costs 123 and 124. The stage-21 controller search is now testing whether the one-action detour changes the loaded carrier’s eight-key route; no assumptions are being made from frame equality.
- The final stage again hit the worker allowance before reporting. To avoid repeating the whole breadth, I’m extracting the single cheapest detour at stage 9, then I can seed only that branch for the remaining route and compare it directly with canonical.
- The stage-9 detour does not survive by simply reverting to every canonical key run; later same-cost substitutions are required. I’m extracting the exact cheapest detour path at stage 20, after which the final two runs can be tested independently without repeating the broad search.
- The extracted state is simpler than expected: the useful detour is a single `1` inserted into stage 20’s six-key load run, producing a distinct post-load carrier configuration at cost +1. I’m seeding exactly that state and searching only the two remaining controller runs for a net saving.
- The stage-20 detour completes in 145 actions, so it is strictly worse. Single-key hidden-state detours are not yielding prefix savings. I’m returning to the global level-7 objective and enumerating alternate peg unload exits on the first wrapped board—routes that could bypass the final relay entirely rather than merely optimize its controller keys.
- The loaded-peg closure is finite but clone-heavy. It is capped at the known 13-key delivery depth and 220 states, and will report every validated physical exit—not just the production landing.
- All 167 loaded-peg controller states at level 7 expose only the reversal and the 13-key production exit; there is no alternate bypass. The accounting and reset avenues are closed. I’m now verifying the claimed global optimum on level 5’s 55-action final region, because if that model is exact, the remaining search can be restricted to its 34-action opening.
- Level 5’s final-region Dijkstra confirms a 55-action global optimum over all capture orders; only its 34-action opening remains, and its fixed-route beam was exact. I’m checking level 8 with the route-wide beam now, including two-key deletions; its synchronized grid has more opportunity for cross-stage carrier alignment than the single-carrier levels.

## Preserved local probes

- `frontier_scaffold.json` (5718 bytes)
- `probe_action7_accounting.py` (355 bytes)
- `probe_action7_capture_undo.py` (1550 bytes)
- `probe_action7_contexts.py` (2877 bytes)
- `probe_action7_shifted_undo.py` (2427 bytes)
- `probe_alternate_controller_paths.py` (3553 bytes)
- `probe_bridge_solver_variant.py` (1843 bytes)
- `probe_capture_suffix_shortcut.py` (5986 bytes)
- `probe_carrier_reset_solver.py` (1703 bytes)
- `probe_carrier_weighted.py` (2842 bytes)
- `probe_coordinate_chaining.py` (1760 bytes)
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
- `probe_l9_bridge_shuttle.py` (5444 bytes)
- `probe_l9_carrier_click_moves.py` (2677 bytes)
- `probe_l9_carrier_pixels.py` (1881 bytes)
- `probe_l9_crossing.py` (6093 bytes)
- `probe_l9_empty_world.py` (1660 bytes)
- `probe_l9_far_clicks.py` (3405 bytes)
- `probe_l9_far_reset_unlock.py` (2654 bytes)
- `probe_l9_far_transfers.py` (2431 bytes)
- `probe_l9_frontier_reset.py` (1935 bytes)
- `probe_l9_frontier_undo_state.py` (1794 bytes)
- `probe_l9_hidden_agents.py` (3594 bytes)
- `probe_l9_hidden_turns.py` (2840 bytes)
- `probe_l9_key_tree.py` (2737 bytes)
- `probe_l9_late_turns.py` (2387 bytes)
- `probe_l9_layout_turns.py` (5570 bytes)
- `probe_l9_loaded_rail.py` (2191 bytes)
- `probe_l9_loads.py` (6953 bytes)
- `probe_l9_local_captures.py` (4817 bytes)
- `probe_l9_nonlocal_moves.py` (6670 bytes)
- `probe_l9_onepeg_finalize.py` (2775 bytes)
- `probe_l9_onepeg_openings.py` (3923 bytes)
- `probe_l9_opening_goals.py` (15173 bytes)
- `probe_l9_opening_mitm.py` (6468 bytes)
- `probe_l9_prereveal_keys.py` (1789 bytes)
- `probe_l9_reset_reveals.py` (1910 bytes)
- `probe_l9_reuse.py` (1463 bytes)
- `probe_l9_selected_keys.py` (2663 bytes)
- `probe_l9_shifted_load.py` (1705 bytes)
- `probe_l9_short_meet.py` (5886 bytes)
- `probe_l9_stage.py` (14974 bytes)
- `probe_l9_variant_model.py` (5685 bytes)
- `probe_l9_world_model.py` (8597 bytes)
- `probe_leg_completion_index.py` (2263 bytes)
- `probe_level_controller_trace.py` (3069 bytes)
- `probe_level_event_closures.py` (6836 bytes)
- `probe_level_legal_moves.py` (2438 bytes)
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
- `probe_route_beam.py` (8092 bytes)
- `probe_shortcut_bridge_level.py` (9559 bytes)
- `probe_stage_click_effect.py` (3161 bytes)
- `probe_transition_undo.py` (1735 bytes)
- `probe_tune_bridge_solver.py` (2242 bytes)
- `probe_undo_goal_search.py` (7259 bytes)
- `probe_verify_integrated_legs.py` (1754 bytes)

Run or inspect the smallest relevant probe before writing another one.
