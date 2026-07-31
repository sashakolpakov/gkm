# Unverified frontier brief: vc33 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace and the documented harness/perception surface. I’ll begin with `solver_index.md`, then reproduce any useful prior observations from `frontier_brief.md` before relying on them, and keep clone probes compact and bounded.
- The frontier records an unverified three-reservoir hypothesis and several bounded probes. I’m treating those only as pointers; next I’m establishing the actual baseline, then I’ll rerun the smallest preserved level-7 composition/probe that can confirm or falsify it.
- The baseline is confirmed at six completed levels. The smallest preserved composition probe is still running; its current search bounds are larger than ideal, so I’m giving it one bounded pass and will tighten or stop it if it doesn’t produce a compact result promptly.
- The preserved composition result falsifies simple reuse as-is: neither the color-15 nor color-14 call changed the level-7 state before the expensive color-11 search. The likely reason, visible in the indexed leg, is that level 7 reuses marker colors for large active gates, while the existing leg requires exactly two components of that color. I’m adding a tiny symbolic probe to verify the object partition and one-step affordances directly.
- The compact probe confirms the world model: action `6` is a coordinate interaction, eight persistent cyan pads transfer fluid by two rows between adjacent reservoirs, and barriers become clickable only at matching pressure. The reproduced three-action sequence opens/crosses the upper-right gate, reducing color 15’s horizontal mismatch from 26.5 to 9.5; the existing alignment leg then reaches zero vertical mismatch. I’m tracing that committed subpath and the resulting active gates so the next relay can be planned symbolically.
- The trace clarifies the dense objective: after crossing color 15 into the central reservoir, the alignment leg places its moving marker at the fixed marker’s row (50). At that moment color 11 is six rows high and color 14 is 24 rows low. Because every pad transfers exactly two rows, I can preserve color 15 by pairing each lower-reservoir fill with an upper-reservoir refill of the center, while independently bringing colors 11 and 14 to their fixed rows. I’m testing that conservation plan directly on a clone.
- The direct conservation test exposed a real barrier constraint: color 11 stops at row 32, so further pressure does not move it; compensating the center only dislodges color 15. That means color 11 must traverse its closed barrier before final height alignment. I’m now testing the existing vertical-gate crossing leg from the verified color-15 staging state, which is exactly the earlier mechanic in a new configuration.
- The existing crossing leg succeeds for color 11 from the staged state: it equalizes pressure, clicks the lower-left gate, moves the platform into the center, and finishes with both color-11 markers at row 26. I’m preserving that verified leg reuse and testing the opposing-direction leg for color 14 from this exact state; color 15’s displacement remains a coupled cleanup subgoal.

## Preserved local probes

- `probe_level4.py` (13644 bytes)
- `probe_level5.py` (4716 bytes)
- `probe_level6.py` (3765 bytes)
- `probe_level7.py` (7566 bytes)
- `probe_level7_best.py` (3241 bytes)
- `probe_level7_color14.py` (1975 bytes)
- `probe_level7_compact.py` (2308 bytes)
- `probe_level7_compose.py` (1533 bytes)
- `probe_level7_plan.py` (1531 bytes)
- `probe_level7_reuse.py` (2334 bytes)
- `probe_level7_score.py` (3289 bytes)
- `probe_level7_search.py` (5641 bytes)
- `probe_level7_trace.py` (2622 bytes)
- `reproduce_level4.py` (5779 bytes)

Run or inspect the smallest relevant probe before writing another one.
