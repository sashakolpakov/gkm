# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 5 at 111 actions.
- Remaining real-action budget under the harness cap: 489 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The directed socket topology is still being exhausted at the bounded rate; it has passed the central placements and is now checking the farther left/right turn positions. No reward has appeared yet.
- The central marked bar is indeed the strongest invariant: in the lower marked level, moving its marker off the projected port breaks an otherwise connected arrangement, and level 6 starts it exactly on both central port axes. I’m holding that bar fixed and closing the remaining cross-product the prior probes split apart: both turn columns and the carrier column, at the exact upper/middle/lower row bands.
- The fixed-marker cross-product is still running normally and has not rewarded in its first portion. It is now in the mixed-side turn placements that neither the central-stack nor fixed-carrier probes jointly covered.
- The fixed-row cross-product is exhausted without reward, so the failure is now in row topology rather than an untested column pairing. I’m switching to a compact categorical sweep over all meaningful side-port rows and left/central/right orderings, with the marked bar held on its verified central column. This covers the remaining directed topologies in about seventeen thousand observational states.
- The categorical sweep is progressing without the overlap-state exceptions after filtering physically colliding placements. It has not rewarded yet; the remaining states cover the alternate turn orientations and carrier orderings.
- Selection is now verified irrelevant: every piece can be active before `USE` on a known winning layout. One exact preserved hole remains after that check—the fixed central marker with upper/lower turns and the vertical carrier below them, but with all three lateral positions varied jointly. I’m closing that cross-product now.
- With the central marker and turn rows fixed, carrier top 14 is exhausted. The adjacent upper-band rows 17 and 20 are still running across all lateral positions; no reward has appeared so far.
- The lower-level acceptance maps sharpen the geometry: target notches are usually met by a movable edge immediately beside them, while source projections may pass through feeder bars. For the bottom notch at columns 29–31, the 15-wide bar’s candidate terminal placements are therefore left 14 (ending at 28) or left 32 (starting at 32), not its initial centered overlap. I’m testing those edge-aligned bottom carriers with the two turns on the source column.
- The edge-aligned bottom-bar family also misses. I’m now checking the complementary low-cost hypothesis: the level starts with all four columns intentionally aligned, and only their vertical ordering is scrambled. This exact-column row sweep was not covered by the representative-column categorical search.
- The rows-initial / columns-only sweep is still running and has not rewarded. It is testing all lateral placements while preserving the observed starting row order exactly.
- The columns-only sweep remains healthy but is slower because each carrier placement is replayed independently to avoid clone-history errors. It has passed the central columns and is finishing the outer placements.
- The `USE` pixel is confirmed to be only an action indicator—there is no hidden partial score. The strongest remaining directed construction is now precise: upper-left turn, one horizontal feeder between turns, right-facing turn, and a vertical carrier spanning rows 26–37. Earlier probes held the feeder column fixed; I’m varying that feeder across every source-compatible column while preserving the left-to-right ordering.

## Preserved local probes

- `probe_axis_ranges.py` (1883 bytes)
- `probe_axis_roles_exhaustive.py` (3479 bytes)
- `probe_categorical_l6.py` (3474 bytes)
- `probe_central_stack.py` (3763 bytes)
- `probe_chain_candidates.py` (1824 bytes)
- `probe_column_sweep.py` (2566 bytes)
- `probe_commit_selection.py` (1066 bytes)
- `probe_connected_cost.py` (5079 bytes)
- `probe_constraints.py` (4143 bytes)
- `probe_edge_carrier.py` (2314 bytes)
- `probe_endpoint_sweep.py` (2572 bytes)
- `probe_exact_l6.py` (2761 bytes)
- `probe_l1_map.py` (979 bytes)
- `probe_l2_acceptance.py` (2569 bytes)
- `probe_l2_rows.py` (1608 bytes)
- `probe_l5_relations.py` (1722 bytes)
- `probe_l6.py` (9720 bytes)
- `probe_l6_conditioned.py` (4068 bytes)
- `probe_l6_connected.py` (2572 bytes)
- `probe_l6_interval_model.py` (8445 bytes)
- `probe_l6_junctions.py` (3442 bytes)
- `probe_l6_map.py` (1076 bytes)
- `probe_l6_ray_model.py` (3321 bytes)
- `probe_l6_ycovers.py` (3287 bytes)
- `probe_local_l6.py` (3628 bytes)
- `probe_marker_control.py` (1321 bytes)
- `probe_middle_a_rows.py` (2353 bytes)
- `probe_middle_rows.py` (2567 bytes)
- `probe_port_chain.py` (2820 bytes)
- `probe_port_shapes.py` (1377 bytes)
- `probe_role_assignments.py` (4411 bytes)
- `probe_root_a_sweep.py` (2165 bytes)
- `probe_row_sweep.py` (2590 bytes)
- `probe_search_l6.py` (4397 bytes)
- `probe_serial_columns.py` (2422 bytes)
- `probe_serial_d_rows.py` (2495 bytes)
- `probe_serial_rows.py` (2480 bytes)
- `probe_snapshots.py` (1466 bytes)
- `probe_socket_permutations.py` (2972 bytes)
- `probe_trace_candidate.py` (1440 bytes)
- `probe_turn_contexts.py` (1421 bytes)
- `probe_turn_rows.py` (1714 bytes)
- `probe_use_status.py` (1455 bytes)

Run or inspect the smallest relevant probe before writing another one.
