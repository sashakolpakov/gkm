# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The branch reaches seven stable slots early, but the beam’s best state remains unable to place middle-right through action 67; the helper is moving away. Two deadline layers remain, so I’m keeping the fixed bound to see whether a lower-ranked avatar route completes the hand placement.
- The two-stage beam reaches a courier-occupied middle slot at 68, but the courier is empty. Inspecting the third-stage map shows why: the block was dropped at row 3, while the courier travels horizontally on row 2 and never collects it. I’m enumerating only the final one-to-three drop actions from the verified carry position to find a cargo placement on the courier’s actual path.
- A useful drop variant emerged: `[up, right, up]` keeps the local block with the avatar at row 1 while changing the courier cycle so middle-right becomes settled automatically. That leaves the avatar 13 turns to hand-deliver top-right. I’m beam-searching this exact turn-56 state with the same corrected dense metric.
- The turn-56 state did not retain a player-carried block—the local cargo has been handed into the courier system. Progress begins only when that courier returns at turn 62. I’m continuing because the branch was specifically selected for an automatic middle-right delivery, but the remaining seven actions must also settle top-middle and top-right.
- This branch can now settle top-right by action 66, but middle-right is still empty at 67. Two final layers remain; a win requires the staged courier to place middle-right at 68 and reward at 69.
- The staged-courier branch also stalls with an empty courier occupying middle-right at 68; the local “stage” did not become a carried delivery. I’m testing the remaining earlier allocation: put the second remote block in top-right before dismissal, then use the local block for middle-right while leaving bottom-middle to the helper. This changes which physical block owns each final slot.
- The remaining untested leverage is during positioning itself: from action 55 the avatar is carrying the local block along the row immediately above the target ring. All prior position sweeps used only up/right moves before interacting. I’m now allowing interactions and direction changes from that exact turn-55 state, with a 400-state beam through action 69.
- Allowing early interaction reaches the target ring sooner: by action 61 the avatar occupies top-right, and by 62 it can occupy middle-right. Neither is settled yet, but seven actions remain for detachment and the second placement.
- Top-right is now settled by action 63, leaving only middle-right and six deadline actions. The beam is exploring the required return/detach sequence; this is the strongest pristine branch so far.
- No middle-right settlement has appeared by action 64. Five actions remain; the search is still within its 25,000-transition cap and keeping 400 unique frames.
- At action 65 the search again exposes the complementary branch: middle-right is settled while top-right holds the empty courier. Four actions remain, but a win still requires a real top-right block rather than occupancy.
- By action 66 the empty courier has left top-right, so the currently best middle-settled branch cannot finish. Three bounded layers remain for lower-ranked player-cargo states; no cap expansion is planned.

## Preserved local probes

- `probe5_candidates.py` (1216 bytes)
- `probe5_structure.py` (1413 bytes)
- `probe8_candidates.py` (4200 bytes)
- `probe8_combo.py` (3142 bytes)
- `probe8_combo_finish.py` (1783 bytes)
- `probe8_current_tail.py` (1896 bytes)
- `probe8_entry.py` (1367 bytes)
- `probe8_fast_finish.py` (1475 bytes)
- `probe8_lower_manual.py` (1541 bytes)
- `probe8_minimize.py` (2312 bytes)
- `probe8_reposition.py` (2565 bytes)
- `probe8_reverse_finish.py` (1688 bytes)
- `probe8_reverse_idle.py` (1498 bytes)
- `probe8_reverse_search.py` (2709 bytes)
- `probe8_reverse_stage.py` (1789 bytes)
- `probe8_third_minimize.py` (750 bytes)
- `probe8_trace.py` (2423 bytes)
- `probe9.py` (13826 bytes)
- `probe9_actual_candidates.py` (4474 bytes)
- `probe9_actual_mutations.py` (2597 bytes)
- `probe9_actual_ports.py` (8832 bytes)
- `probe9_alt.py` (2466 bytes)
- `probe9_below_pick.py` (2167 bytes)
- `probe9_best_mutations.py` (3689 bytes)
- `probe9_best_short_verify.py` (1522 bytes)
- `probe9_best_trace.py` (1409 bytes)
- `probe9_candidate.py` (3993 bytes)
- `probe9_combined_variants.py` (1477 bytes)
- `probe9_context.py` (1760 bytes)
- `probe9_courier.py` (1563 bytes)
- `probe9_current.py` (19683 bytes)
- `probe9_delay_dismiss.py` (1711 bytes)
- `probe9_delete_one.py` (4252 bytes)
- `probe9_entry.py` (2521 bytes)
- `probe9_exact_finish_search.py` (2766 bytes)
- `probe9_fast_suffix.py` (2641 bytes)
- `probe9_handoff.py` (1261 bytes)
- `probe9_idle_initial.py` (803 bytes)
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
- `probe_minimize_segments.py` (3388 bytes)
- `probe_pair_minimize.py` (4700 bytes)
- `probe_solver_timing.py` (980 bytes)
- `wip_glue_notes.md` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
