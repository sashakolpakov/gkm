# Unverified frontier brief: sb26 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with `solver_index.md`, then use only the documented harness/perception surface and compact clone probes.
- The baseline reliably completes level 1 in 9 moves. I’m now inspecting only the indexed function ranges plus the public harness entry points needed to clone the level-2 state and summarize it symbolically.
- Level 2 is again coordinate-driven (`5` submit, `6` click), with seven colored swatches at both top and bottom and a central two-row structure. I’m testing the existing color-code leg on a clone first, as required, before inventing any new mechanic.
- Click probes confirm the mechanic: select a bottom swatch, then click a central placeholder to color it. There are seven placeholders plus one prefilled `14`, so this is not the earlier direct seven-slot copy; I’m testing a small set of geometry-based orderings and using completion reward as the verifier.
- The straightforward row and snake readings all fail, so the diagram’s ordering is genuinely nontrivial. I’m switching to a bounded clean-room search over only the seven visible placeholders and seven visible swatches: 13,700 partial assignments (far below the scaffold’s 20,000-state default), cloning each partial state and verifying only with reward.
- The bounded search is behaving as expected and has checked 1,000 of at most 5,040 complete arrangements without touching the real level-2 state.

## Preserved local probes

- `probe_level2.py` (4310 bytes)
- `search_level2.py` (1722 bytes)

Run or inspect the smallest relevant probe before writing another one.
