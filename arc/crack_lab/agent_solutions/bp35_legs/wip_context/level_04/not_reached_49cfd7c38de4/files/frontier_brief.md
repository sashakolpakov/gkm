# Unverified frontier brief: bp35 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with `solver_index.md`, then reproduce any useful frontier observations through bounded clone probes before editing the level-3 player.
- Baseline is confirmed at level 3 after 60 real moves. The frontier’s mechanic claims are still untrusted; I’m now inspecting the smallest preserved level-3 probe and the narrow API/leg ranges needed to reproduce them symbolically.
- The reproduced start establishes the avatar and controls: action 3/4 moves the colour-9/11 body left/right; action 6 is coordinate interaction. It also verifies two distinct click affordances: colour 14 disappears, while colour 12 can either disappear or expand depending on its shape/context. I’m testing the preserved ten-band opening and a tightened symbolic search from that verified state.
- The opening is reproduced exactly: clearing the colour-12 gate lets the avatar cross; clearing the overhead colour-14 block gains four bands; staged sideways movement then gains six more. The symbolic search is still inside its 3,000-state bound and has not yet produced a result, so I’m letting this bounded run finish before tightening it further.
- The broad search’s useful partial path reaches 15 bands, but its branching model is too expensive. The next phase confirms the puzzle’s timing constraint: each crossing requires neutralizing only nearby small colour-12 hazards, and wasting clicks reaches a global hazard phase around move 30. I’m replacing the broad frontier with a local-action search keyed by cell shape, avatar column, accumulated height, and move count.
- The tightened search found and clone-verified a 34-action completion at 249 expansions, with dense progress of 21 climbed bands before the final prize approach. I’m promoting that search policy—not the one-off path—into a reusable local hazard-climb leg, then wiring level 3 to compose only that leg and replay-validating the full solver.
- Full validation passes: `RESULT levels=3 moves=94 replay_ok=True err=None`. I’m doing the final narrow cleanup now—removing the temporary probe-only search body and checking the edited solver files compile—while preserving the reusable leg and level dispatch.
- Implemented level 3 using reusable hazard-climbing legs. - Added shape-aware, bounded local hazard search in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_bp35_budgeted_medium_screen/legs.py:373). - Added `play_level_3` as a leg-only composition in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_bp35_budgeted_medium_screen/players.py:16). - Validation: `RESULT levels=3 moves=94 replay_ok=True err=None`. - Syntax checks pass.

## Preserved local probes

- `frontier_scaffold.json` (1700 bytes)
- `l1_probe.py` (7220 bytes)
- `l1_search.py` (2988 bytes)
- `p3.py` (1634 bytes)
- `probe.py` (7635 bytes)
- `probes3.py` (9594 bytes)

Run or inspect the smallest relevant probe before writing another one.
