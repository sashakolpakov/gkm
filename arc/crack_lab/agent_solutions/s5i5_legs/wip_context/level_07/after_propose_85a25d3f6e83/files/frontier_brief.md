# Unverified frontier brief: s5i5 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and the documented `gkm_try.py`/`perception.py` surface. I’ll begin with `solver_index.md`, then the frontier brief if present, establish the level-7 state with `gkm_try.py`, and use compact clone probes before editing the leg library and player.
- The baseline is clean and reproducible: six levels complete in 173 moves. I’m now inspecting only the indexed player/leg ranges and the smallest preserved level-7 probe, then I’ll rerun that probe against a fresh clone so none of the frontier claims are taken on trust.
- Fresh probing confirms two coupled assemblies and the useful progress signal: the upper marker is already docked at `(7,22)`, while the lower four-link endpoint must reach the hollow center `(16,25)`. The lower endpoint’s direct motion is blocked by partitions, so I’m tracing link pivots—not just endpoint distance—to identify the pose needed to carry the final pivot through the corridor.
- The first bounded global search is still running but has not produced a better state yet. Its old heuristic only rewards endpoint distance and tends to rediscover the same trapped pose, so in parallel with monitoring it I’m refining the state score around the corridor requirement: first place the penultimate pivot in columns `30–32`, then rotate/extend the terminal link toward `(16,25)`.
- The corridor geometry now yields a concrete subgoal: the terminal joint must end at `(16,31)` with its last link pointing left. To make that possible, the preceding joint must first be translated from about `(31,43)` into the three-cell corridor. I’m testing the corresponding fold–retract–translate–rotate choreography on a clone before committing it to a reusable leg.
- The four-link arm reaches the corridor, but every lower-arm control then locks before the final six-cell horizontal offset. One unverified interaction remains: the isolated color-8 control is inert in the baseline context but may unlock after the already-docked upper slider is displaced. I’m checking that contextual action now, since it is the only control not represented in the lower-arm search and could provide the required handoff inside the left chamber.
- The contextual probe was meaningful: moving the upper slider two steps does unlock the isolated rotation, but it only reorients the separate color-8 arm and does not change the lower corridor mechanics. I’m returning to a bounded global search now with the verified lower-arm choreography as a strong prefix; the remaining search must allow a temporary retreat because every direct move from the six-cell pose is mechanically locked.
- The bounded search is progressing normally at roughly 2,000 novel configurations per minute and remains below the requested clone rate. It has exhausted the easy direct-distance states; the next useful evidence will be either a reward path or the first configuration that moves the lower endpoint inside column 31 while preserving the upper dock.
- At 13,000 distinct states, the search has reproduced both known dense-progress poses—`(31,25)` and `(16,31)`—but no reward yet. That confirms the six-cell trap is not a probe artifact; the solution needs a state whose endpoint distance is temporarily worse, which this pass is now explicitly expanding.
- The pass has crossed 21,000 configurations without exceeding its 30,000-state cap. No closer marker state exists in this explored region, but the frontier is still growing rather than exhausting, so I’m letting the bounded pass complete before tightening the search around a different joint pose.

## Preserved local probes

- `probe_level2.py` (2922 bytes)
- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `probe_level5.py` (3516 bytes)
- `probe_level6.py` (3311 bytes)
- `probe_level7.py` (5222 bytes)
- `probe_level7_ascii.py` (1273 bytes)
- `probe_level7_choreo.py` (1933 bytes)
- `probe_level7_frontier.py` (1909 bytes)
- `probe_level7_stage2.py` (2026 bytes)
- `probe_level7_trace.py` (1402 bytes)
- `probe_level7_turns.py` (2344 bytes)
- `probe_level7_upper.py` (1444 bytes)
- `search_level4.py` (2167 bytes)
- `search_level5.py` (3054 bytes)
- `search_level6.py` (2339 bytes)
- `search_level7.py` (3022 bytes)
- `search_level7_local.py` (3314 bytes)
- `search_level7_pivots.py` (3980 bytes)

Run or inspect the smallest relevant probe before writing another one.
