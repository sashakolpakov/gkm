# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The type-swapped path does not finish, so cargo identity matters later. I’m comparing native and swapped states at each remaining segment boundary to locate the first semantic divergence—likely when the remote peg appears—before deciding whether the swapped route offers a cheaper alternate capture.
- The native five-move edge sequence is not valid under the swapped carrier phase; divergence occurs immediately, before the remote peg. I’m treating the 40-action swapped endpoint as its own state and enumerating its legal relay moves rather than reusing incompatible coordinates.
- The swapped endpoint’s carrier graph is being enumerated. I’m looking for legal forward moves of both cargo pieces; their shortest paths will determine whether this alternate phase is viable or a dead end.
- No opportunity list yet; the state remains within the 1,000-state cap. I’ll allow the bounded scan one more interval.
- The swapped endpoint scan is slower than the prior 124-state graphs, suggesting a substantially larger phase. I’m waiting for its terminal opportunity list rather than extending depth further.
- The swapped endpoint is viable: it reveals a new phase with the peg at `(30,24)` and an immediate bridge jump `(54,12)→(54,24)`. I’m following that unique forward move and re-enumerating; this is now a distinct, empirically valid cargo-order route rather than a splice.
- After the immediate swapped bridge jump, the carrier graph is again being searched for the next legal cargo move. A short path that brings the bridge toward the peg would make the alternate order competitive.
- No opportunity list yet; the scan remains within the same bounded depth/state limits. I’m waiting for its terminal result.
- The alternate phase is clone-bound but still bounded to 15 key steps. I’ll allow one final interval for the opportunity list.
- The scan has exceeded the usual interval without result. I’m stopping it rather than let a neutral-cost alternate consume the remaining probe budget; bridge-first has not yet demonstrated any action saving.
- I’m narrowing the swapped bridge state to a 10-key horizon, matching the native transition scale. If no forward cargo move exists there, the alternate ordering is already slower than the route it would need to beat.
- Within the 10-key horizon, the swapped bridge has an immediate forward jump `(54,24)→(54,36)`; no keys are needed. I’m following it once more. Up to this point the alternate has exactly matched the native action cost, so the next phase must produce a strict shortcut to remain useful.

## Preserved local probes

- `campaign_candidate_633.json` (11978 bytes)
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
- `probe_leg_reuse.py` (3154 bytes)
- `probe_level9_abstract_entry.py` (3812 bytes)
- `probe_level9_context_actions.py` (2272 bytes)
- `probe_level9_direct_finish.py` (3235 bytes)
- `probe_level9_entry_variants.py` (6196 bytes)
- `probe_level9_reproduce.py` (4363 bytes)
- `probe_level9_stages.py` (2238 bytes)
- `probe_prefix_suffix_matches.py` (6854 bytes)
- `probe_preselected_carrier.py` (1665 bytes)
- `probe_selected_actions.py` (2149 bytes)
- `probe_suffix_splices.py` (4611 bytes)
- `probe_verify_best_campaign.py` (1518 bytes)
- `probe_verify_campaign_candidate.py` (922 bytes)
- `wip_glue_notes.md` (11184 bytes)

Run or inspect the smallest relevant probe before writing another one.
