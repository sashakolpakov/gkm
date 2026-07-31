# Unverified frontier brief: s5i5 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed surface only. I’ll first establish the current level, then reproduce any prior brief observations on fresh clones before relying on them.
- Baseline is reproduced at six completed levels. The brief’s level-7 geometry is still untrusted; I’m now checking the smallest preserved level-7 probe and the harness transition path, then I’ll rerun compact measurements from that exact frontier.
- Fresh probing confirms the useful dense measure: the controllable lower marker begins at `(16,55)` and its hollow target is centered at `(16,25)`. The direct route is blocked by the chamber walls, while controls independently translate and reorient linked segments. I’ve started the bounded symbolic search from the validated checkpoint, keyed only on the gameplay region.
- The search found a much stronger reproducible route, reducing marker distance `30 → 6` and returning it to the target row at `(16,31)`. It stalls because the attached arm still spans a partition; I’m probing rotations at that exact state to identify the final fold, rather than expanding the costly general search.
- At `(16,31)`, the apparent “horizontal” control now moves vertically—orientation matters, and the arm can be lifted to `(10,31)` but cannot fold there. A rotation sends it below the lower partition, confirming a multi-stage reorientation puzzle. I’m switching to a bounded local search rooted at the six-cell frontier so the remaining exploration does not repeatedly rediscover the solved first half.
- The local search has explored over 5,000 distinct arm configurations without losing the established prefix; none yet improves on distance six. That is useful evidence that the final maneuver must temporarily move away from the target. I’m keeping the cap bounded and allowing those detours while preserving the already-docked upper marker.
- The six-cell frontier is blocked even after moving the upper slider aside, so that object is not the missing clearance. The remaining issue is the lower arm’s internal pose before it enters the narrow opening. I’m comparing the proven three-link choreography from level 6 with level 7’s extra control pair, then I’ll probe only the corresponding staged sequence.
- Including the upper slider does not open an immediate final move: another 5,000 configurations still bottoms out at distance six. I’m letting this last bounded pass finish; if it exhausts, I’ll use its evidence to stage the lower links earlier in the route and test the level-6-style fold directly.
- The trace exposes the bottleneck precisely: the final rotation places the endpoint’s pivot at about `(30,44)`, so the horizontal fold can only reach column 31. The hollow target requires that pivot six cells farther left before the fold. I’m probing controls immediately before that rotation to shift the pivot through the three-cell corridor, which is the dense subgoal the earlier search was missing.
- A second staged route now threads the endpoint through the corridor while keeping the next link below the partition. It reaches the same `(16,31)` distance but in a genuinely different arm pose, so I’m searching locally from this pose; this is the configuration that can plausibly bring the next pivot through instead of trapping it on the right.

## Preserved local probes

- `probe_level2.py` (2922 bytes)
- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `probe_level5.py` (3516 bytes)
- `probe_level6.py` (3311 bytes)
- `probe_level7.py` (5222 bytes)
- `probe_level7_frontier.py` (1909 bytes)
- `probe_level7_trace.py` (1374 bytes)
- `search_level4.py` (2167 bytes)
- `search_level5.py` (3054 bytes)
- `search_level6.py` (2339 bytes)
- `search_level7.py` (3022 bytes)
- `search_level7_local.py` (3314 bytes)

Run or inspect the smallest relevant probe before writing another one.
