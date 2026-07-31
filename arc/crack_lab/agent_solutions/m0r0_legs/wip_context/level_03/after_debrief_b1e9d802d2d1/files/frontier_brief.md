# Unverified frontier brief: m0r0 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with `solver_index.md`, then use only the documented harness/perception surface and compact clone probes.
- The baseline reliably reaches level 3’s start (`levels=2`, 38 moves). I’m now inspecting only the indexed helper definitions and the small public harness entry points needed to create a level-3 clone.
- Level 3 again contains two independently positioned color-10 objects, and actions 1–4 move both every turn while 5/6 do nothing at the start. That strongly matches the earlier mirrored-pair mechanic; I’m reducing the frame to a symbolic cell map and tracking the two objects under each action before planning.
- The apparent “walls” are actually the connected color-5 track: both avatars move on it, while the large left/right color fields are barriers. The three tiny color-9 marks lie on that track and are plausible staged endpoints. I’m testing reachability with reward as the real goal and avatar-to-mark distance as the dense diagnostic.
- The bounded search is still running; it has not produced a result within the first 30 seconds. I’m letting this single capped search finish, then I’ll inspect only its compact path result.
- Both trials stop exactly one cell from the marker—markers are solid terminals, not floor tiles. No interaction action changes anything when only one avatar is touching one terminal. The next concrete test is simultaneous contact with the left and right terminals, which is a much better dense subgoal than raw frame change.
- Simultaneous side-terminal contact is unreachable, so that hypothesis is ruled out. I’m now comparing the exact pre-reward and post-reward frames of the two already-verified levels; this should identify whether reward comes from avatar reunion, terminal contact, or another invariant without guessing.

## Preserved local probes

- `l3_observe.py` (3963 bytes)
- `probe_level3.py` (7448 bytes)

Run or inspect the smallest relevant probe before writing another one.
