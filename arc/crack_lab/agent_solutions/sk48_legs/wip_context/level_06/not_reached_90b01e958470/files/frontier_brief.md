# Unverified frontier brief: sk48 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API surface, then reproduce any inherited observations before probing level 6. I’ll keep the player purely compositional and validate the final solver through `gkm_try.py`.
- The baseline run is still starting up due to a one-time local plotting cache build; no game observations have been consumed yet. Once it finishes, I’ll isolate level 6 on clones and summarize it symbolically rather than dumping frames.
- The baseline confirms the current solver cleanly completes five levels. I’m now constructing a fresh level-6 state through the public harness, then testing every advertised action on independent clones to identify the avatar and context-free effects before planning.
- The probes identify the mechanics cleanly: actions 1/2 move the left avatar by one lane, 3/4 retract/extend its horizontal segmented tether, and 6 is inert even after movement/extension contexts. Level 6 is a cooperative geometry: the left collector requests the three 9s, while a fixed top collector requests the three 8s; a central solid block prevents either initial straight-line collection. I’m testing a bounded observational search next, using completion reward as the terminal check and token-to-collector staging as the dense interpretation.
- The first breadth-first pass has not completed within a minute, which means raw frame BFS is exploring too many equivalent tether-length states. I’m stopping that pass and will quotient the state by logical lane/token positions so the next search remains bounded and interpretable.
- The logical heuristic is now steering toward both requested trains, but clone reconstruction is the bottleneck; the search is still within its 12k-state cap. I’ll give this bounded pass one more short interval, then either use its candidate or reduce it to staged subgoals rather than expanding the cap.
- The dense subgoal probes have now exposed the crucial distinction: a token stays on the horizontal tether after being routed, unless it is captured by the fixed collector. I’m testing action 6 at exactly that attached-at-target context; its initial no-op made it easy to misclassify, and this is the context-dependent affordance the raw action set suggests.

## Preserved local probes

- `probe_l6.py` (2049 bytes)
- `search_l6.py` (2122 bytes)

Run or inspect the smallest relevant probe before writing another one.
