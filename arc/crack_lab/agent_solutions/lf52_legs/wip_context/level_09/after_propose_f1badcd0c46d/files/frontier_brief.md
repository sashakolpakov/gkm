# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The alternative carrier route is geometrically valid: move the carrier to column 54, relay the peg and bridge pair through columns 42 and 48, then load at 54. This costs six extra actions but should shift the phase-2 carrier alignment by 12 pixels—the exact phase discrepancy observed. I’m verifying the alternative phase transition now.
- At offset 5 the pegs align with exactly one empty slot between them. Although ordinary pegs cannot step without jumping, the remote object may be a cooperative agent on this phase. I’m testing the dense subgoal of moving it directly from `(12,22)` to `(24,22)`; if allowed, the loaded peg can immediately capture through it.
- The carrier’s left motion has been blocked by the movable bridge at `(36,16)`. Candidate phase-1 ending #1 places another bridge at `(30,16)`, allowing the blocker to jump upward to `(24,16)` while the peg stays loaded. I’m testing left immediately after clearing that obstruction; this directly applies the reachability constraint.
- Clearing only row 36 did not unlock the carrier. Candidate #1 can also clear the row-30 bridge by moving it to row 18; I’m testing up after both obstructions are removed, which is the last distinct reachability context for the loaded carrier.
- The bridge network may route the hidden peg rather than merely support jumps. I’m testing that directly: at the junction, relocate the right bridge from column 28 to 16, then reverse six conveyor steps until the remote peg re-enters. A changed row or column would confirm routing and provide the missing capture path.
- The junction rearrangement persists through reversal, confirming those bridges are routing state. Before abandoning the junction, I’m completing its only two-step normalization: move bridges from `(22,28)` to `(16,22)`, then to `(10,16)`. This tests whether the terminal condition includes staging the bridge pair after transporting the remote peg.
- The wrap logic suggests the junction bridges are a gate: action 4 was tested at the stop before moving them, but not after. I’m relocating the blocking bridge from `(18,28)` to `(18,16)`, then pressing right once. If the conveyor advances, that opens the next region where the transported peg can reappear.
- Moving one junction bridge does not release right. I’m completing both bridge moves and testing down in that new configuration; this is a genuinely new key context and the most likely turn into the next wrapped segment.
- The candidate-0 junction is a dead end under every contextual key. I’m comparing the alternate staged phase-1 ending at the same conveyor stop; its different bridge rows should wrap to a different junction topology, which can change reachability without adding phase-2 actions.
- The junction also supports the opposite staging `(28,34)`. Right remains blocked there; I’m testing down in this alternate configuration, which can select a different wrapped branch than the normalized-left pair.
- The alternate junction branch is also inert. I have not found a replay-verified terminal suffix, so I’m removing the exploratory scripts and avoiding an unverified player that would consume the remaining move budget. I’ll leave the established level 1–8 solver intact and run the harness once more to confirm no regression.
- I couldn’t produce a replay-verified level-9 solution. Final harness result remains: `RESULT levels=8 moves=544 replay_ok=True err=None` All exploratory files and unverified changes were removed; the existing level 1–8 solver remains intact.

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
- `probe_level9_splice_suffixes.py` (2517 bytes)
- `probe_level9_stages.py` (2238 bytes)
- `probe_level9_undo_edges.py` (1992 bytes)
- `probe_prefix_suffix_matches.py` (6854 bytes)
- `probe_preselected_carrier.py` (1665 bytes)
- `probe_selected_actions.py` (2149 bytes)
- `probe_suffix_splices.py` (4611 bytes)
- `probe_verify_best_campaign.py` (1518 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (17952 bytes)

Run or inspect the smallest relevant probe before writing another one.
