# Unverified frontier brief: bp35 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with the indexed API map, then reproduce any useful prior observations before probing level 3 symbolically. I’ll keep all committed level logic as compositions of legs.
- The baseline reliably clears two levels in 60 moves. The preserved brief contains no facts, only probe pointers, so I’m reproducing level-3 observations from the smallest existing scripts before relying on them.
- Level 3 reuses left/right movement and coordinate interaction, but introduces colour-12 objects: clicking them removes them, while the three top-row instances mutate rather than simply behaving like prizes. The immediate dense-progress route is clear—remove the colour-12 gate, cross right, and clear an overhead block to gain three bands—so I’m testing that subgoal and the resulting top configuration.
- The key new mechanic is now verified: small colour-12 spikes are lethal, while clicking expands them into solid, safe barriers; full barriers can be clicked down to become passable. That makes level 3 a timed “set the cells before crossing” climb rather than an ordinary removable-block tower. I’ve reached 10 bands of verified progress and am switching to a bounded clone search over left/right plus only the currently visible interactive cells to resolve the remaining phase timing.
- The bounded search is still running; its first cost is dominated by exact visual-state keys, not environment stepping. I’m letting this bounded run finish, then I’ll tighten the key to symbolic cell state if needed before making solver changes.

## Preserved local probes

- `frontier_scaffold.json` (1700 bytes)
- `l1_probe.py` (7220 bytes)
- `l1_search.py` (2988 bytes)
- `p3.py` (1634 bytes)
- `probe.py` (7635 bytes)
- `probes3.py` (9594 bytes)

Run or inspect the smallest relevant probe before writing another one.
