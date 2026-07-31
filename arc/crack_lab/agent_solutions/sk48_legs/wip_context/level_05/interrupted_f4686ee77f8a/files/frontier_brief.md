# Unverified frontier brief: sk48 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The refined dense measure found the correct avatar alignment in only three actions: `retract×2, up` places the avatar on the same lane as the visible `8-9-8`. Reward still does not fire, so alignment alone is insufficient; the remaining subgoal is a genuine horizontal tether connection to that row across the central barrier.
- Composing two existing legs exposed a much cleaner near-state: the controlled prefix is an `8` parked just left of the barrier, while the far side is `9-9-8`. That reduces the finish to a standard one-joint ladder maneuver: lift the controlled `8` above the barrier, align it over one unwanted `9`, and descend to peel that `9` out. I’m verifying each possible alignment on clones now.
- The ordinary-ladder clone probe ruled out a one-step descent: any tether still spanning the central block makes vertical movement collide with the barrier. I’ve therefore seeded the search from the cleaner `8 | 9-9-8` frontier and am preserving staged joints; one malformed off-board branch exposed by the probe is now skipped locally rather than treated as progress.
- The diversified search found a stronger ladder result: it cleanly peels the extra `9`, leaving the avatar aligned with exactly `8-9-8` and parking all three surplus tokens on other lanes. The reward remains absent, so I’m continuing only from configurations that preserve those surplus-token separations while seeking the final tether attachment.
- The earlier-level reward trace clarifies the real win condition: level 1 advances exactly when an already controlled ordered prefix extends into the final token. So for level 5 the right dense signal is not “a visible `898` row”; it is the longest `8`, then `8-9`, sequence that actually shifts under retract/extend. I’m retargeting the bounded search to that experimentally verified control signal.
- The new control-based search has now reached the decisive subgoal: a genuinely tether-controlled `8-9` prefix sits left of the barrier, while the nearest token on the far side is the required final `8`. The other tokens are staged away. I’m keeping the fixed search bound focused on routing that two-token prefix across the barrier and making the final contact.
- The controlled-pair state matches the earlier verified three-token ladder geometry, just with one fewer carried token. I’m testing the same proven ladder skeleton with only its train-length-dependent spans varied (`retract`, middle shift, and far reach); this is a much smaller symbolic probe than continuing unrestricted search.
- Existing-leg composition found a more useful bridge than the raw ladder replay: `weave` followed by `unweave` places the controlled `8-9` pair entirely above the far side at columns 36/42, directly over the far `8-9-8` row. I’m seeding the finish search there; this preserves the proven reusable operations and reduces the remaining geometry to the final descent/contact.
- The adapted ladder has now produced verified physical progress beyond the barrier: it pushes the nearest required `8` down to the clear lower lane while leaving an `8` as the controlled prefix. That creates the correct retrieval geometry—controlled `8`, available `9`, then the lowered final `8`—so I’m seeding the control search from this transfer rather than from the blocked far-side row.
- Repeating the already-verified collection maneuver from the lowered-token state moves the required `8` all the way around the barrier to the left side at `(19,12)`. A `9` is on the same clear upper lane at `(19,36)`. I’m now searching from this reproduced retrieval state for the short horizontal collection that turns those into the controlled `8-9` prefix.
- The retrieved-left state is stable, but the default weave controls the wrong color (`9`). I’m sweeping only the two documented weave parameters on clones to align the avatar with the retrieved `8`; this tests reuse of the existing general leg before introducing any new level-specific primitive.
- The retrieval now suggests a repeatable cycle: bring one required token left of the barrier, collect the next token with it on the clear upper lane, retract the `8-9` pair to columns 12/18, then run the verified lowering route for the final `8`. I’m validating that composed cycle stepwise before committing it as a reusable leg.

## Preserved local probes

- `probe_level5.py` (10202 bytes)
- `probe_level5_candidate.py` (2018 bytes)
- `probe_level5_coordinate.py` (2751 bytes)
- `probe_level5_empty_below.py` (5705 bytes)
- `probe_level5_empty_collect.py` (2899 bytes)
- `probe_level5_finish_legs.py` (3380 bytes)
- `probe_level5_ladder.py` (1509 bytes)
- `probe_level5_pair_ladder.py` (8260 bytes)
- `probe_level5_routes.py` (4998 bytes)
- `probe_level5_strong.py` (2333 bytes)
- `probe_level5_transfer_path.py` (1280 bytes)
- `probe_level5_vertical.py` (2251 bytes)
- `probe_prior_mechanics.py` (1887 bytes)
- `search_level5.py` (4089 bytes)
- `search_level5_bfs.py` (1408 bytes)
- `search_level5_eight.py` (2902 bytes)
- `search_level5_finish.py` (7075 bytes)
- `search_level5_random.py` (3164 bytes)
- `search_level5_transfer.py` (3602 bytes)

Run or inspect the smallest relevant probe before writing another one.
