# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API surface. I’ll reproduce any frontier observations I use, probe level 8 symbolically on clones, then keep `players.py` as composition-only and verify with `gkm_try.py`.
- Baseline reproduced: level 7, 133 moves, replay valid. The preserved level-8 notes are still only hypotheses; I’m now checking the smallest relevant probes and the harness entry points, then I’ll rerun the decisive action/object experiments rather than trusting the brief.
- The action experiment is now reproduced. A click on the 7×7 color-8 square moves it to a reachable clicked center; clicks elsewhere still advance three color-7 bodies diagonally, confirming a mixed direct-control/autonomous system. Four corner rings are persistent targets. I’m letting the bounded dense search finish, using assignment distance to the four rings rather than raw frame change as progress.
- The first search confirmed the dense metric is meaningful: it staged the color-8 square in the upper-left ring and brought the three moving bodies to within a total Chebyshev distance of 5 from the remaining rings. The broad action generator then exhausted its shallow frontier. I’m narrowing from that reproduced near-state to the actual control affordance—clicking a body pauses it while the others advance—and searching only those four meaningful choices.
- The narrowed search is still running; it has only four branches per state and remains within the clone-rate budget, but output is buffered until completion. The key now tracks only the staged square and three body centers, so this is testing timing states rather than pixel-level noise.
- The near-state probe exposed the missing mechanic: a body is not a single “pause” target. Clicking different color-7 pixels on the same 8-pixel body produces different diagonal jumps/orientations; 24 colored pixels gave 24 meaningful outcomes. The earlier center-only key collapsed those directional states. I’m updating the bounded search to preserve each body’s exact pixel signature and use only currently clickable body pixels.
- The orientation-aware search has not yet beaten distance 5, but it is exploring distinct exact-pixel states now rather than revisiting collapsed centers. I’m keeping it bounded at 3,000 expanded states; if it does not produce reward, I’ll use its best frontier to derive the ring-entry condition directly.
- The symbolic ring patches clarify the win condition. The square and two bodies are fully contained by three ring masks; the remaining body straddles the upper-left edge of the lower-right ring. That gives a sharper dense metric: count controlled/body pixels occupying original ring cells. I’m switching from center distance to this exact containment score, which should reward the final fit even when body centers have coarse parity.
- The containment metric is working: the beam has improved from 21/24 body pixels inside ring masks to 23/24 while keeping the square fully staged. The required state appears to be exact 24/24 simultaneous containment; I’m continuing the bounded timing search from this frontier.
- The 40-state beam reached 23/24 containment but eventually cycled, so the goal is close but the best-only frontier is too narrow: synchronizing periodic movers requires temporary loss of overlap. I’m preserving small quotas across every overlap score, which keeps controlled detours while staying bounded, and will capture the first 24/24/reward path.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_beam8.py` (3293 bytes)
- `probe_dots.py` (903 bytes)
- `probe_finish8.py` (4381 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level8.py` (2994 bytes)
- `probe_patterns8.py` (913 bytes)
- `probe_root8_actions.py` (1824 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
