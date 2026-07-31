# Unverified frontier brief: cd82 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then run the current solver and make compact, clone-only probes for level 3 before editing the leg library and player dispatch.
- The baseline is clean: levels 1–2 complete in 11 moves, with replay validation passing. I’m now reading only the indexed solver definitions and the public harness/API entry points needed to obtain a level-3 clone and summarize it symbolically.
- The third stage is exposed as `levels_completed == 2` (the harness’s final result counts completed stages). Its scene has a top selector strip and a bounded central work object. Directional actions 3/4 move a persistent object; action 5 changes the work region, while 1/2/6 are inert in the initial context apart from a status pixel. I’m tightening that into color-transition and region summaries, then I’ll test actions in multiple positions.
- The canvas is a 10×10 target-composition puzzle. The useful dense measure is exact target-cell matches (0 initially): applying a stamp paints a deterministic half-plane mask, and moving the vessel changes that mask (top half, side half, or a diagonal). Palette clicks only change paint color. I’ve verified these effects across multiple colors and positions; next I’m enumerating the small vessel-position graph and searching compositions against the 100-cell target.
- A useful correction emerged: the 10×10 reference is not achievable by naïvely treating every application as a final paint layer—each available mask covers at least 50 cells, while no final reference color occupies that many. I’m checking the already-validated level-2 leg against its pre-completion reference to identify the actual comparison rule, then I’ll use reward-bounded search rather than assuming pixel equality.
- The reference geometry now identifies a compact recipe search: the F region uniquely requires the northwest mask; the C, E, and 8 regions each fit only a few vessel masks. That reduces the plausible four-stamp recipes to 576 ordered candidates, small enough to test directly on clones with the real reward instead of extrapolating paint behavior.

## Preserved local probes

- `l3_compose_search.py` (1487 bytes)
- `l3_context_probe.py` (1937 bytes)
- `l3_masks.py` (1087 bytes)
- `l3_observe.py` (1601 bytes)
- `level_refs.py` (748 bytes)
- `probe_candidate.py` (2663 bytes)
- `probe_history.py` (932 bytes)
- `probe_level3.py` (1644 bytes)
- `probe_plan.py` (1730 bytes)
- `probe_search.py` (2997 bytes)
- `probe_stamps.py` (1144 bytes)

Run or inspect the smallest relevant probe before writing another one.
