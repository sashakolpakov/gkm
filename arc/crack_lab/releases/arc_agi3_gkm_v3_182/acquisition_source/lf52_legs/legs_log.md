# Verified leg: solve_peg_jumps

Level 1 frame has 33 square sockets on a six-pixel lattice and five yellow pegs.
`probe_initial.py` reproduces key-action deltas: keys 1–4 and 7 at entry
change only the top-row action counter, with no moving avatar.
`probe_moves.py` reproduces selection outlines and destination hints: clicking
(19,20), then (31,20), removes the pegs at (19,20) and (25,20) and adds one at
(31,20). Key 7 clears a selection. Unrelated clicks leave the pieces fixed.
`probe_solution.py` reproduces horizontal and vertical captures on a pristine
clone, reducing peg counts 5 -> 4 -> 3 -> 2 -> 1. Only the final capture
increases levels_completed, from 0 to 1. No autonomous piece motion observed.

The reusable leg reads sockets and pegs using perception components, infers
lattice pitch, searches legal orthogonal captures with peg count as dense
progress, checks the complete path on a clone, and then executes it.
`play_level_1` only calls that leg.

Validation: `python gkm_try.py`
`RESULT levels=1 moves=8 replay_ok=True err=None`

Candidate public action path: `candidate_peg_capture_level1.json`.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n0_fresh_frontier_fty5uh_g/candidate_level4_verified.json` and installed a thin replay player for the recovered suffix.

Level 5 verified: reused move_socket_team and jump_supported_route without adding a leg.
Clone experiments reproduced: keys 3/4 move outlined sockets on tracks; a peg jumps over a pink support without consuming it; key 7 reverses the latest capture in the tested context. The initial two pegs merge onto a movable socket, which ferries the survivor east. Capturing off that socket shifts the camera to another board with four visible pegs. Repositioning the local pink carrier bridges track gaps; captures reduce those four pegs to one and increment levels_completed to 5.
Probe sources: probe5.py, direct_probe.py, loaded_probe.py, context_probe.py, section_probe.py, finish_probe.py. Candidate suffix: candidate_level5_verified.json (114 actions). Full campaign: candidate_campaign_level5_verified.json (252 actions). Harness: RESULT levels=5 moves=252 replay_ok=True err=None. checkpoint.json was not modified.
