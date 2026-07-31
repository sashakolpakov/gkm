# Unverified frontier brief: lf52 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed summaries. Then I’ll reproduce any useful prior observations through bounded clone probes before changing the level-6 player.
- The frontier suggests a partially mapped carrier-and-bridge variant, but I’ll treat it only as a list of hypotheses. I’m now running the current solver to establish the actual level boundary and then I’ll use the preserved compact probe to reproduce the mechanics.
- The baseline is confirmed at five completed levels. The compact level-6 run is still spending most of its time replaying the first five levels, so I’m letting that finish once; afterward I’ll checkpoint the staged state in-memory within each probe and keep each experiment short.
- The preserved probe was unsuitable because it rebuilt the earlier search-heavy levels. I’ve stopped it and will use the validated prefix exactly as the local harness does, keeping all new observations independent and compact. I’ll avoid broad workspace searches from here and inspect only named source ranges or probes.
- The initial frame now reproduces two directional mechanics: the color-8 piece leaps over a peg without removing it, while a peg can leap over color 8 and the latter persists. I’m using those verified rules in a small symbolic search, with peg count as dense progress, and will replay the resulting macros on a clone before relying on them.
- The dense-progress sequence is verified: it reduces three local pegs to one, loads that peg, and then loads the color-8 avatar beside it. Nine right inputs reveal the second region. There, keys change context as expected—up moves the avatar off the carrier, while horizontal travel before unloading scrolls the carrier/world. I’m tracing the avatar’s corridor from that staged endpoint with short branches.
- The first right-edge loop was too shallow: it returned the avatar to the same side of the lower board, so no legal leap appeared. The observations show the camera can finish scrolling while the avatar remains off-screen; I’m testing a few longer excursions to distinguish “camera travel” from actual movement around the barrier, rather than expanding an opaque breadth-first search.

## Preserved local probes

- `probe_level6.py` (1119 bytes)
- `probe_level6_macros.py` (13153 bytes)
- `probe_level6_search.py` (5292 bytes)

Run or inspect the smallest relevant probe before writing another one.
