# Unverified frontier brief: s5i5 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace boundary, begin with `solver_index.md`, and use only the documented harness/perception surface. I’ll first confirm the current level, then build compact clone probes for level 2 and implement the solution as composed legs.
- The harness confirms level 1 is already solved in 13 clicks. Level 2 is coordinate-only (`action 6`), so movement-key probing does not apply; I’m advancing a controlled run to level 2, summarizing its components, and testing one click per candidate object on clones to identify the interactive pieces and a dense progress signal.
- The level is a constrained locomotion puzzle, not another slider alignment: a one-cell marker on the articulated colored body must reach the hollow four-cell target at `(31,50)`, while a vertical wall blocks the direct route. The four paired controls extend/retract differently colored limbs. I’m now measuring which controls translate the body marker and how retraction behaves after extension, then I’ll search only over these eight symbolic controls.

## Preserved local probes

- `probe_level2.py` (2922 bytes)

Run or inspect the smallest relevant probe before writing another one.
