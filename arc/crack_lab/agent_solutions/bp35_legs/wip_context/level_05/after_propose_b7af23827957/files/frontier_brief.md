# Unverified frontier brief: bp35 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The one-time setup replay remains the dominant cost; the cache is now local to `/tmp`, so later probe processes won’t rebuild fonts. I’m keeping this run alive because it combines setup, bounded planning, and clone reward validation in one clean lineage.
- The first bounded planner confirmed the current leg cannot solve level 5 within 1,200 observational states and depth 64. I’m now mapping exactly where its affordance model stops: phase-by-phase band shifts, repeated safe left movement from the fourth chamber, and only clicks whose effects are terminal, camera-shifting, or larger than a local block deletion.
- The map isolated the missing affordance. In chamber four, the avatar can safely move from column 5 to 3, but entering column 2 kills it beneath two cells whose colors are outside the current gravity leg’s model. I’m testing coordinate action 6 on those exact cells from different distances and with repeated clicks, using “survives the next left step” as the subgoal.
- This probe varies two independent conditions: avatar distance from the unknown cells and one/two/three repeated interactions. That will distinguish a local shape-toggle from a globally removable block and from a mere no-op whose only visible change is the move counter.
- The experiment identifies color 12 as a reversible local hazard: from the adjacent column, one or three clicks make the next left crossing survivable, while zero or two clicks kill the avatar. This exactly matches the earlier shape-toggle affordance, so I’m generalizing the existing gravity-room leg to treat both observed hazard colors (12 and 15) as locally interactable, then I’ll rerun the clone planner before composing level 5.
- The generalized search is now running on a fresh level-5 clone with the same 1,200-state/64-action bound. A successful result must both return a path and replay that path to `levels_completed == 5`; otherwise I’ll inspect the next frontier rather than increasing bounds blindly.
- Setup replay has completed slowly but deterministically in prior runs; the actual planner previously took under ten seconds. I’m preserving the single-process run because it gives the needed clean clone validation without committing any level-5 action to the real solver.
- The new affordance lets the avatar pass the first color-12 hazard, but the full BFS still exhausts its bound without reward. I’m advancing manually by verified subgoals now: toggle the hazard directly above the intended column, cross once, repeat for the neighboring hazard, and measure the resulting camera shift before probing the next chamber.
- The bounded experiment is checking only the two staged hazard crossings and the immediate local neighborhood afterward. This keeps the probe causal: each camera shift is tied to one verified interaction/move sequence, and distant block deletions are excluded from the next affordance model.
- The verified color-12 crossings reveal a five-band upward transition into the next chamber. The remaining failure is search order: plain breadth-first spends its state budget on reversible local states before following camera-height progress. I’m changing the same reusable gravity leg to best-first search on cumulative band shift, while retaining reward as the only success test and the same context-safe action set.
- The dense-progress planner is now being validated on a clone. Its priority is cumulative camera ascent, but it will only return when the real level reward increases; terminal states and states where the avatar disappears are still discarded.
- Height-prioritized search still exhausts the same bound, so another affordance is missing. The next chamber has the avatar under four solid ceiling cells and a removable support above column 4; I’m verifying the natural subgoal—walk under that aligned support and remove it—then mapping the newly exposed frontier.

## Preserved local probes

- `probe_level5.py` (1382 bytes)
- `probe_level5_after12.py` (3043 bytes)
- `probe_level5_contexts.py` (2781 bytes)
- `probe_level5_deeper.py` (2917 bytes)
- `probe_level5_map.py` (2909 bytes)
- `probe_level5_search.py` (1035 bytes)
- `probe_level5_unknown.py` (2295 bytes)

Run or inspect the smallest relevant probe before writing another one.
