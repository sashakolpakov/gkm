# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin from `solver_index.md`, then reproduce any frontier observations I rely on with bounded symbolic probes. I’ll inspect only the documented harness/API surface and the listed source ranges.
- The baseline is confirmed at five completed levels. The frontier notes point to a preserved level-6 probe, but its claims are unverified; I’ll inspect its compact logic and the relevant player/leg definitions, then rerun only the smallest measurements needed to establish the board and action mechanics.
- The mechanics are now reproduced: action 6 selects the clicked movable piece, actions 1–4 translate it by three pixels, and action 5 tests/commits the arrangement. Level 6 has four jointly relevant pieces—two mirrored L shapes, a vertical bar, and a horizontal bar with an embedded marker—plus three side sockets and one bottom socket. I’m tightening the goal model by perturbing known winning arrangements from levels 4–5, which should reveal the exact projection constraints more efficiently than blind search.
- The structured row-chain candidates still do not complete level 6, so the missing constraint is not just socket-ray coverage. The level-5 perturbations show a sharper fact: orthogonal translations can be irrelevant for some pieces, while one-grid shifts along the constrained axis break completion. I’m now identifying which pieces and coordinates are actually essential in a known level-4 win, then I’ll use that reduced constraint pattern to bound the level-6 search.
- The reduced search is running at a capped 280 action steps per second. It enumerates only socket-aligned rows, keeps the horizontal marked bar at its observed column, and tests the two plausible sides for the vertical bar; this is 2,744 final arrangements rather than an unconstrained four-piece sweep.
- The first staged run exposed an instability from long-lived clone histories while sweeping a piece through many positions. I replaced that inner sweep with short, independently replayed clones; the candidate space is unchanged, remains inside the documented action surface, and is still rate-limited.
- The expanded sweep is still progressing normally. It covers 6,912 final arrangements built around the two plausible long-bar spans and both endpoint/flanking placements of the L pieces; no output is expected until a reward hit or exhaustion.
- The search remains within its bounded budget and action-rate cap. If this socket-aligned family exhausts without reward, the evidence will isolate the missing degree of freedom to the marked horizontal bar’s column or to non-endpoint row placements, rather than the already-covered L/bar flank choices.
- No completion signal yet, but the run is healthy. The direct-clone variant costs more replay actions than the earlier snake walk, so this bounded pass takes several minutes; it is testing exact reward after every arrangement, not relying on frame-change heuristics.
- The 6,912-arrangement family exhausted without reward. That rules out the simple “marked bar fixed, long bar spans adjacent sockets, L pieces flank it” model. Before expanding another axis blindly, I’m mapping the full one-piece tolerance intervals around the known level-5 win; those intervals reveal whether the goal uses coverage, adjacency, ordering, or exact endpoints.

## Preserved local probes

- `probe_constraints.py` (3616 bytes)
- `probe_l6.py` (9720 bytes)
- `probe_search_l6.py` (4428 bytes)

Run or inspect the smallest relevant probe before writing another one.
