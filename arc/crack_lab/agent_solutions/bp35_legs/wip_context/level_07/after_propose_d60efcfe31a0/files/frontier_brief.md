# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with the indexed API map, then reproduce any prior observations I rely on before changing the level-7 player.
- The baseline reliably reaches level 7’s start (`levels=6`, replay valid). There is no prior frontier brief, so I’m treating level 7 as fresh and will inspect only the harness/API entry points plus the narrow existing-leg ranges needed to construct bounded probes.
- The initial probe is still reconstructing the six-level prefix; once it reaches level 7 it will report only the 10×8 symbolic lattice and action-local deltas, keeping the raw output compact.
- The reproduced mechanics match the existing gravity/support family: keys 3/4 move the avatar, coordinate action 6 changes color-12 support shapes, and color-15 cells are non-interactive hazards. The level’s opening constraint is spatial: the avatar starts in a lower corridor and must approach the nearest support before it becomes clickable. I’ll now test the existing context-safe gravity-room search unchanged on a level-7 clone.
- The unrestricted variant confirms that “click everything” is too broad for the existing search and creates an expensive state fan-out. I’m narrowing experimentally: first measure the support’s within-cell shape state, then test approach → toggle → movement sequences around the opening corridor. This should identify the actual affordance without brute force.
- A new affordance is visible outside the terrain lattice: ten color-8 controls form a vertical strip, one aligned with each world row. This explains why terrain-cell-only search cannot solve the layout. I’ll probe each row control independently and measure which support cells it changes.
- The left strip is a lethal runtime-state display, not a control; a single click there destroys the avatar, so I’m excluding it. The useful missing context is narrower: the next support sits two lattice rows below the avatar, just outside the existing search’s one-cell interaction radius. I’ll test a two-cell support radius—small enough to avoid the prior combinatorial blow-up.
- The coordinate within an object does not change its effect: each color-12 object has exactly two shapes (area 5 ↔ 21), while color-15 is inert. Before assuming traversal is the goal, I’m checking the other plausible dense objective directly on a clone: expand every color-12 object once and see whether completion—not mere frame change—fires.
- A consistent physical explanation now fits the evidence: the color-8 strip is a global gravity control, and clicking it at the start kills because the avatar falls without a safe landing. The lower color-12 support can be expanded first. I’m testing that staged pair now—support first, gravity flip second—which directly mirrors the cooperation mechanic from level 6.

## Preserved local probes

- `probe_level7.py` (5089 bytes)

Run or inspect the smallest relevant probe before writing another one.
