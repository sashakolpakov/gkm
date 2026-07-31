# Unverified frontier brief: sp80 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and the exposed harness/perception surface. I’ll begin with `solver_index.md`, inspect only its referenced ranges, then run the baseline and use bounded symbolic clone probes before editing.
- Baseline is clean at level 4 (`79` moves, replay-valid). The existing levels establish a select–drive–commit family, so I’m now advancing a harness clone to level 5 and will summarize only colors, blobs, and action deltas—not raw frames.
- The level-5 frame has four selectable pieces: three bars (widths 9, 12, 15) and one distinct 6×6 piece. The arrow keys move whichever piece is selected by 3 pixels; USE alone does nothing. Four static color-11 sockets sit on three top edges and one left edge, which points to a two-dimensional alignment variant of the earlier family. I’m testing that hypothesis through piece/socket shape signatures and clone-only placement search.

## Preserved local probes

- `probe_level5.py` (5404 bytes)

Run or inspect the smallest relevant probe before writing another one.
