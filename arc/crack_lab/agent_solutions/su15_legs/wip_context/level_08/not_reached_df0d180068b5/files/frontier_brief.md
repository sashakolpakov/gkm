# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The center control is a one-turn hold, not a persistent stop. The reproduced motion is nevertheless highly structured: unselected bodies advance four pixels along their stored heading; directional controls move farther and set that heading; boundary encounters reflect. I’m using that observed model only to propose short timing sequences, then replaying every candidate on a real clone and accepting only an actual reward increase.
- Clone replay exposed a correction to the provisional model: the large immediate displacement from a directional click is reliable, but the body’s later autonomous drift is not simply that clicked vector—ring contact can constrain or redirect components, and visible position alone is insufficient. I’m isolating “click now” from “drift next turn” with two-step controls before resuming search.
- A structured local search around the 23/24 suffix tested 459 one-edit variants. None won, but many distinct variants still end at 23/24, including reordered controls; those carry different latent motion states. I’m now expanding every such 23/24 endpoint by all 27 body controls before considering a wider mutation search.
- The 22 distinct 23/24 latent endpoints have now all been expanded one turn; their best successor is 22/24, confirming none is one move from completion. The combined endpoint BFS is evaluating the full two-turn frontier now, still capped at roughly 300 clone steps per second.
- No two-turn continuation wins; the best transient state there is 16/24, so the solution must deliberately move farther away before converging. I’ve retained 1,200 behaviorally diverse two-turn states (current frame plus two prior body layouts and last control) for the third-turn expansion.
- The third-turn expansion is larger than the first two but remains within the requested clone-rate cap. It has not triggered reward early; I’m letting this bounded layer finish so its best reachable overlap can guide whether to continue forward or change the staging prefix.
- Deep clone retention became the bottleneck during the third layer, so I stopped that branch without changing the real environment. I’m switching to the scaffold’s path-only replay pattern: retain compact action paths, rebuild bounded candidates from the staged root, and discard each clone immediately.
- A stronger inconsistency surfaced from the existing reusable leg semantics: color-7 bodies were previously treated as moving cutters, and the staged prefix converts each color-11 square into lower-mass fragments—the exact signature of cutter damage—not legitimate delivery. Since that prefix never earned reward, I’m discarding its ring-overlap theory and reproducing a winning earlier cutter level to identify the real invariant before planning level 8.
- The earlier-level reward check refines the picture: level 4 rewards merging all square mass and delivering the final square, while level 6—the reusable pattern relevant here—has two rings and is won by staging a 7×7 square in one ring, then advancing a separate partner into the other. I’m inspecting that winning transition’s exact partner/ring overlap to confirm whether level 8 is the four-ring generalization.
- Level 6 confirms the relevant goal signal: the 7×7 square is staged first, then reward follows when the 8-pixel partner reaches full ring overlap. Level 8 is therefore the four-ring version; the lower-color squares are collision byproducts, not the final targets. I’m adding a verified ring-click “wait” context to the timing mutations, since prior searches only redirected or held a body.
- Repeated neutral waits do not improve the 21/24 staged state and eventually terminate, so timing must be actively controlled. I’m moving to a bounded evolutionary search over short control programs, seeded by the verified 23/24 path; each candidate is replayed from the same staged clone, scored by exact ring pixels and assignment distance, and checked against real reward on every step.
- The evolutionary replay has reached an exact 24/24 body-ring state after eight controls—the first full dense-goal state reproduced. Reward is delayed, as in level 6, so the search is now extending that exact state to find the safe confirming action that increments `levels_completed`.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_beam8.py` (3293 bytes)
- `probe_beam_clean8.py` (4425 bytes)
- `probe_bfs23.py` (2174 bytes)
- `probe_body_controls8.py` (1396 bytes)
- `probe_clean8.py` (5740 bytes)
- `probe_cutter_goal.py` (1919 bytes)
- `probe_direction_persistence8.py` (1228 bytes)
- `probe_dots.py` (903 bytes)
- `probe_endpoint_bfs8.py` (5827 bytes)
- `probe_evolve8.py` (6704 bytes)
- `probe_exact_search8.py` (4032 bytes)
- `probe_finish8.py` (4381 bytes)
- `probe_frontier23.py` (3109 bytes)
- `probe_hold8.py` (1788 bytes)
- `probe_hold_persistence8.py` (1214 bytes)
- `probe_hold_search8.py` (5274 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level6_detail.py` (1973 bytes)
- `probe_level8.py` (2994 bytes)
- `probe_model8.py` (5969 bytes)
- `probe_patterns8.py` (913 bytes)
- `probe_root8_actions.py` (1824 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `probe_state8.py` (1031 bytes)
- `probe_symbolic8.py` (1517 bytes)
- `probe_variants8.py` (5313 bytes)
- `probe_waits8.py` (2108 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
