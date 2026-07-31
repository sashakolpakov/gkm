# Unverified frontier brief: bp35 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin from `solver_index.md`, then use only the documented harness/perception surface and compact clone probes.
- Baseline is clean at three completed levels. The frontier brief contains only level-3 history, so I’ll now reproduce level 4 directly by advancing a clone with the existing players and summarize objects/action effects symbolically.
- The level-4 probe is still advancing through the validated level-2/3 searches; I’m letting that bounded setup finish once, then I’ll preserve the resulting replay prefix so later experiments won’t repeatedly pay that cost.
- The first dense-progress probe climbs five bands by walking right, then hits a complete ceiling while a color-15 object sits beyond it. That rules out treating level 4 as a plain climb; I’m now testing the context-dependent possibilities the frame suggests—pushing the two left-side objects, removing their supports first, and interacting with the upper object/ceiling after the ascent.
- Clicking the upper color-15 object is now ruled out as progress: it produces a death/fall state, and exhaustive move/visible-click exploration closes after only 68 observational states. The raw action list also names action 7 even though this level’s `env.actions` omits it, so I’m testing that otherwise inert action again in the only materially different contexts—at the ceiling, on the hazard, and on removable blocks.
- The ceiling object is color 8, not the color-15 hazard class. Clicking it is therefore a verified state transition, not a generic hazard click; my first search omitted it by over-narrowing interactive colors. I’m correcting the action generator to click every visible off-terrain lattice object and rerunning the bounded state search.

## Preserved local probes

- `frontier_scaffold.json` (1700 bytes)
- `l1_probe.py` (7220 bytes)
- `l1_search.py` (2988 bytes)
- `level4_prefix.json` (702 bytes)
- `p3.py` (1634 bytes)
- `probe.py` (7635 bytes)
- `probe4.py` (5322 bytes)
- `probes3.py` (9594 bytes)
- `search4.py` (2299 bytes)

Run or inspect the smallest relevant probe before writing another one.
