# Unverified frontier brief: sc25 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin from the provided indexes, then reproduce only the observations needed with bounded clone probes. I’ll keep level-specific dispatch thin and put any reusable behavior in `legs.py`.
- The baseline reliably reaches level 4’s start (`levels=3`, meaning three completed). I’m now isolating the level-4 frame and testing each action on clones, first for the moving avatar and then for contextual effects.
- The avatar is the 4×4 two-colour object at rows 19–22, columns 35–38; actions 1–4 translate it by one object-width. Action 6 is contextual: at the default centre it flips the four orthogonal cells of the visible 3×3 panel. I’m treating “remaining lit panel cells” as the first dense progress measure and searching that small state space before tackling the separated movement regions.
- Reaching the small colour-14 marker with two down and two right moves reproducibly removes it, reducing the marker count by four without completing the level. That validates a movement subgoal. I’m now probing the two reachable boundary interactions from that state: the green object on the left and the blue gate/object on the right.
- Pure navigation exhausts the symbolic movement state graph without a reward, so the panel is necessary. Its nine samples are all colour 2 in this configuration, which makes the existing colour-selection leg directly applicable; I’m verifying whether selecting that entire class is the missing subgoal and whether marker collection must precede it.
- A compact raw crop resolves the panel logic: the reference diagram marks the four orthogonal positions around the centre. The no-coordinate form of action 6 reproducibly exposes exactly those four panel cells as colour 0, matching the existing “select cells of colour” skill. I’m testing that verified composition both before and after collecting the movement marker.
- Four coordinate selections constitute one evaluated panel command: the panel resets, and the remote right-hand agent changes shape, confirming the panel controls another actor across an unreachable barrier. I’m enumerating the bounded 4-of-9 command set on clones (126 cases) after marker collection, using reward plus the remote agent’s symbolic shape as the outcome.
- The enumeration isolates exactly one effective command: the four-cell reference cross. It shrinks the controllable avatar from 4×4 to 2×2 while leaving the remote actor unchanged. That explains the earlier blocked corridor: this is a verified “resize to pass” mechanic. I’m now searching the newly reachable movement graph under the remaining turn budget.
- The resized avatar can reach the otherwise inaccessible left portal via one down move and sixteen left moves; direct collision is blocked. The upper diagrams now supply the next contextual command: the portal’s green reference is a three-cell vertical line. I’m verifying that command while adjacent, which fits the remaining turn budget exactly.
- The vertical command opens both formerly sealed side gates. The small avatar can enter the left portal but cannot finish alone, while the large right-hand actor remains outside the central region. I’m testing the remaining coordinate affordance directly on that actor—selection followed by movement—to verify the expected control handoff rather than assuming autonomous motion.

## Preserved local probes

- `probe.py` (11441 bytes)

Run or inspect the smallest relevant probe before writing another one.
