# Unverified frontier brief: sk48 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then reproduce any useful frontier observations with compact clone probes before editing the level-6 composition.
- The frontier contains only hypotheses, so I’m treating them as unverified. I’m running the required baseline now; afterward I’ll inspect only the narrow harness/API ranges needed to create a level-6 clone and reproduce the action/object observations.
- The reproduced frame shows four selectable collectors, not one: clicking an endpoint with action 6 transfers control, and the same directional keys then act geometrically—along a tether they extend/retract it; perpendicular directions move the whole train. I’m now testing the four obvious straight collections independently and together, using token-component counts and reward as the dense/terminal measures.
- The first inherited staging macro is valid and informative: the horizontal collector pushes the nearest 8 right, pulls it left of the obstacle, raises it above the obstacle, then places it directly under the vertical collector. That is a verified dense subgoal, but the token remains attached to the horizontal tether. I’m testing the handoff now—selection plus vertical retraction/extension—to learn exactly when ownership transfers before routing the remaining pieces.
- The handoff has one more timing constraint: with an existing collected token, extending the target train and retracting the staging tether must be interleaved; my first repeated macro placed the later tokens but did not transfer ownership. Rather than guess through the remaining collision states, I’m switching to a bounded symbolic best-first search from the verified one-token state, keyed by frame plus selected collector and scored by exact distance of the three live 8s to the requested slots.
- The bounded search is still running without a candidate after its first minute; it is exploring clone states rather than increasing the cap. I’ll let this fixed 8k-state pass finish once, then use its best structural result or tighten the state/action abstraction—no unbounded expansion.
- The bounded clone search was too slow because deep clones copy growing replay history, so I stopped it rather than exceed the budget. Focused probes found the exact two-token transfer sequence. The final token is blocked by the center square during that same maneuver; I’m testing the natural cooperative fix now: shift the receiving vertical train one lane left, perform the handoff beside the obstacle, then shift the completed train back onto its requested column.
- The symmetric probes now place all three 9s exactly in their requested row, confirming the routing macros themselves. The remaining issue is the last attachment at the obstacle. I’m testing a compact cooperative handoff: move the two-token receiving train one lane above the blocker, pull the staged third token into that clear lane with the other collector, join it there, then return the completed train to its goal row.

## Preserved local probes

- `probe_l6.py` (2049 bytes)
- `probe_l6_paths.py` (7983 bytes)
- `search_l6.py` (2122 bytes)
- `search_l6_coop.py` (3283 bytes)

Run or inspect the smallest relevant probe before writing another one.
