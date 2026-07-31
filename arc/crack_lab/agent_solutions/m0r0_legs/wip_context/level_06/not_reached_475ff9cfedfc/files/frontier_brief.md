# Unverified frontier brief: m0r0 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The baseline is clean and reproducible: 5 levels, 182 moves, replay valid. Next I’m inspecting only the indexed solver ranges plus the documented harness API, then I’ll generate a compact level-6 observation (counts, components, and one-step deltas) without dumping the raw grid.
- The one-step experiment identifies the avatar: the two color-10 4×4 bodies move consistently and mirror horizontally; actions 1/2 are vertical, 3 moves them apart, 4 moves them together, while 5 and unparameterized 6 are inert here. I’m now mapping the barriers at 4×4 resolution and using pair separation as the dense progress signal, while checking whether contacts with the colored bodies alter reachability.
- A bounded key-only search exhausts the observable movement state without reward, so reunion is not achievable by the main pair alone. The stationary color-12/14 bodies split the two reachable regions, matching the earlier selectable-blocker mechanic; I’m testing coordinate selection on each candidate and then each movement key to establish exactly which groups are controllable and how they move.
- The selected 2×2 body can traverse the central lower passage and is physically blocked on the left, but can be parked to the right. This gives a verified decomposition already represented in the library: relocate the selectable corridor blocker, reselect the mirrored pair, then reunite them. I’m searching only the key route for the final reunion across a few verified parking positions.
- The blocker is correctly parked and the main pair is correctly reselected, but the key-only search still cannot pass the paired 12/14 structures. Their geometry leaves no 4-pixel route around them, so I’m testing contact and action 5 immediately adjacent to each colored barrier; this is the remaining context-dependent affordance before planning the route.
- Contact is the switch: stepping onto the paired top 12/14 cells removes both 12-cell and 14-cell central barrier groups. That sharply improves the dense reachability measure—the pair can now descend into the reunion corridor. I’m verifying the shortest sequence with and without first parking the tiny selectable blocker.
- The apparent “disappearance” is contextual and reverses when the pair leaves the colored contact; meanwhile colors 6/7 acquire a deterministic color-0 crack pixel as turns advance. That turn clock was inflating the observational BFS key. I’m now repeating the bounded search with a verified symbolic key that preserves avatars, selection state, switches, barriers, and walls while quotienting out only the irrelevant crack-age pixels.
- The search confirms this is not a simple avatar-reunion level. Contacting the paired top cells starts two autonomous cracks at opposite corners; subsequent turns advance them, so the meaningful dense progress signal is color-0 crack length, not avatar separation. I’m measuring propagation while the pair holds the contacts versus after it leaves, and watching for barrier interactions and reward.
- The crack is a deadline (the run terminates around 140 idle turns), not the goal. The cooperative route is now clear geometrically: while the main pair remains on the two top contacts and keeps the large barriers open, the tiny selected agent can move right out of the central slot, then climb the newly opened right chamber toward the stationary pair. I’m testing that staged assembly and its exact contact depth.
- The tiny agent reaches the right avatar but cannot merge by collision; it can, however, replace an avatar on a pressure contact after selection switches. I’m switching from hand enumeration to a bounded coordinate-aware state search: movement keys plus clicks only on currently visible agent candidates, with crack age quotienting. This directly explores the required handoffs without assuming which agent must hold which switch.
- The coordinate-aware search is still running within its 60,000-state cap; no boundary expansion is involved. Its branching comes from legitimate agent handoffs, so I’m letting this bounded pass finish before narrowing around the best reachable subgoal.
- The full handoff search is slower than the harness target because retained deep clones become expensive. I’m stopping at the existing bounded cap if it does not finish imminently; the next pass will use compact replay paths and a smaller subgoal frontier, which keeps the same clean observation surface.

## Preserved local probes

- `probe_clock.py` (990 bytes)
- `probe_coop.py` (804 bytes)
- `probe_coord_bfs.py` (1447 bytes)
- `probe_l6.py` (3836 bytes)
- `probe_routes.py` (1556 bytes)
- `probe_search.py` (998 bytes)

Run or inspect the smallest relevant probe before writing another one.
