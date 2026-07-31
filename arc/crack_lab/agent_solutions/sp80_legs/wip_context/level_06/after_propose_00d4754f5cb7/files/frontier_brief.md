# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with `solver_index.md`, then check `frontier_brief.md` if present. I’ll keep observations compact and use only the documented harness/perception surface.
- There’s no preserved frontier brief, so this is a fresh clean attempt. The index shows level 6 is entirely unimplemented; I’m running the harness once now to establish the exact starting state.
- The existing five levels confirm one reusable family: coordinate-select an object, drive it with keys, then commit. I’ll now construct a small level-6 observer that replays the existing players, reports only color/component summaries, and compares each action on clones—no raw frame dumps.
- Level 6 is another spatial arrangement puzzle, but with a richer board: four socket-like color-11 targets, several movable color-8/9 pieces, and one color-15 piece. The first clone tests show keys move the already-selected piece; next I’m testing coordinate selection across each candidate and measuring which component responds, including whether color 15 is autonomous or selectable.
- The shapes clarify the objective family: level 6 has three side-facing sockets and one bottom socket, while the movable set is two mirrored 3-cell L pieces, one 4-cell vertical bar, and one 5-cell horizontal bar. This matches level 5’s projection/alignment mechanic rather than one-piece-per-socket coverage. I’m mapping the chamber boundaries and reproducing the already-validated level-5 alignment to extract the exact row/column constraints before searching level 6.
- The validated earlier levels show that completion depends on collective row/column projections, not physical socket overlap. I’ve defined a dense four-ray coverage score (three side openings plus the bottom opening) and am searching only vertical placements first, with clone actions rate-limited to about 300 steps/s.
- Vertical placement alone reached full ray coverage but never completed in 600 trials, so the bottom socket imposes a genuine column constraint too. I’m expanding to joint row/column placements while geometrically reserving each later piece’s selection point, so overlapping pieces cannot corrupt the experiment.
- The joint search is still running within the step-rate cap; no completion has appeared in the first minute. The search is deterministic and preserving exact destination/path data so any hit can be replayed directly as a declarative `drive_objects` plan.
- No sparse-reward hit yet, although full four-ray coverage is common; that tells us the dense measure is necessary but not sufficient. I’m letting this bounded sweep finish before tightening the measure from the unsuccessful configurations.
- The completed earlier levels reveal a stronger dense invariant: every winning arrangement makes the movable pieces’ projections into continuous overlapping chains along each socket axis. Level 4 needs only a column chain; level 5 needs both row and column chains. Level 6 initially has a broken row chain and an isolated vertical bar in columns, so I’m testing the minimal chain-forming arrangement now.
- The first chain attempt compressed everything into rows 14–28, so it did not actually span the three side-socket openings. I’ve corrected the dense target: a continuous row projection from the first opening (23) through the last (40), plus a continuous column projection through the bottom opening. This yields a much smaller, structured candidate family.

## Preserved local probes

- `probe_l6.py` (9720 bytes)

Run or inspect the smallest relevant probe before writing another one.
