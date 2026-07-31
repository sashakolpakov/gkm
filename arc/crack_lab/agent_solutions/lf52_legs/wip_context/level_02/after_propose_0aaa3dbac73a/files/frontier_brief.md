# Unverified frontier brief: lf52 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace and use only the documented harness/perception surface. I’ll begin from `solver_index.md`, then run the baseline and inspect only the indexed ranges needed for level 2.
- Baseline confirms level 1 is solved in 8 moves and dispatch stops because `play_level_2` is absent. Next I’m reading the compact perception helpers and current leg/player definitions, then I’ll create a bounded symbolic probe that reaches level 2 on a clone and summarizes objects/action deltas.
- The first symbolic frame shows two peg lattices plus a separate movable object: a bordered 4×4 core near `(33,31)` responds to left/right, while up/down are blocked in its starting context. I’m now mapping its reachable positions by clone BFS and tracking peg-state changes separately; this will reveal whether level 2 is a navigation/cooperation layer around the familiar peg mechanic.
- Movement BFS found 19 reachable cells but no peg displacement and no reward, so the core is the avatar and the pegs are not pushable. The top lattice itself has a clear three-capture dense-progress path (4 pegs → 3 → 2 → 1), while the disconnected lower lattice already contains one peg. I’m verifying those coordinate interactions on a clone before generalizing the existing peg leg to handle multiple disconnected boards.
- The coordinate sequence is valid: each capture reduces the top board, and completing it recolors the two surviving pegs from 14 to 2. That is verified partial progress, but not the level reward. Completion also reveals a new color-15 object in a separate bottom-left region, consistent with a staged second agent; I’m now testing how each key action affects that object and whether it can reach the remaining lower peg.
- The color-15 object is a large outlined control/figure, and clicking it resets the peg state; it is not the reward target. Since the peg solution is unique, the remaining context variable is the avatar’s location when the board is completed. I’m testing all 19 reachable avatar positions against that verified three-capture solution, then will search the post-solve movement graph only for any distinct resulting states.

## Preserved local probes

- `probe_level2.py` (6693 bytes)

Run or inspect the smallest relevant probe before writing another one.
