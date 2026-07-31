# Unverified frontier brief: lf52 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and the documented harness surface. I’ll begin with `solver_index.md`, then reproduce any prior observations from `frontier_brief.md` before relying on them, and keep probes compact and clone-bounded.
- The brief is only a hypothesis index; I won’t trust its paths or mechanics without replaying them. I’m running the required baseline now, then I’ll inspect only the indexed entry points and the smallest preserved level-5 probes needed to recreate the setup.
- Fresh reproduction confirms level 5 begins with two pegs in separated boards and two bridge structures; left/right move the upper bordered assembly, while up/down and coordinate action 6 are initially inert except for the turn indicator. The existing level-4 leg exposes one legal bridge transfer, so I’m replaying the preserved decision sequence only as an experiment and measuring its resulting symbolic state, not adopting it as a solution.
- The capture sequence reproduces a real dense milestone: peg count drops from two to one. Reward does not fire because that peg remains inside a movable carrier. I also reproduced that raw coordinate clicks no longer select it, so the remaining mechanic is track navigation. I’m mapping only the post-capture carrier/bridge configurations and their reachable junctions now.
- The junction map reveals an important interaction: when the empty bridge assembly is routed through the far-right board edge, a second peg becomes exposed, so that route undoes dense progress rather than completing the level. The filled carrier and bridge share controls but can be pinned independently by walls; I’m bounding the raw-key search to confirm whether navigation alone can ever reward before adding coordinate-transfer macros back in.
- The raw-key graph is finite (678 physical states) and contains no reward, confirming that navigation must expose another legal peg transfer rather than finish directly. I’m scanning those states for symbolic bridge/capture opportunities; after that the solver can alternate “find an alignment” and “perform one coordinate macro,” which matches the reusable level-4 pattern.
- The newly exposed capture is valid, but it transports the view into a larger continuation with four visible pegs; level 5 is therefore the same bridge/carrier peg-solitaire mechanic repeated across a larger world, not a one-off endgame. That strongly favors reusing the existing general bridge-carrier leg. I’m testing that composition before adding any new leg.
- The existing leg’s 120-state alignment cap is too small for level 5; raising it reaches a much larger search and is still running, so I’m treating this as a fit test rather than committing to brute force. If it fails or stalls, I’ll add a minimal reusable variant that keeps compact paths and prioritizes captures by the dense peg-count measure.
- The larger continuation has no immediate transfer, which is exactly where the old leg’s replay-heavy alignment search becomes costly. I’m switching the probe to a direct-node, stop-at-first-alignment search and committing only verified macro stages inside the disposable run; this preserves the same general mechanic while avoiding repeated deep replays.
- The iterative probe is reproducing the mechanics cleanly: two bridge transfers preserve the dense peg count, then a capture reduces 2→1. It is now searching the finite post-capture navigation graph for the next exposed capture; this is the only expensive stage, and it stops at the first verified opportunity.
- The direct search reached a genuine branch: after reducing the larger board from four visible pegs to three, the greedy “first bridge” choice dead-ends in a 74-state component. I’m isolating that junction and testing each legal bridge branch on clones; this is a planning choice, not a reason to broaden the action model.

## Preserved local probes

- `probe_fresh_l5.py` (2038 bytes)
- `probe_l5.py` (3536 bytes)
- `probe_l5_endgame.py` (3166 bytes)
- `probe_l5_finish.py` (1270 bytes)
- `probe_l5_iterative.py` (2804 bytes)
- `probe_l5_key_bfs.py` (1081 bytes)
- `probe_l5_macro_bfs.py` (3365 bytes)
- `probe_l5_raw_frontier.py` (1919 bytes)
- `probe_l5_tracks.py` (3671 bytes)
- `probe_level2.py` (6693 bytes)

Run or inspect the smallest relevant probe before writing another one.
