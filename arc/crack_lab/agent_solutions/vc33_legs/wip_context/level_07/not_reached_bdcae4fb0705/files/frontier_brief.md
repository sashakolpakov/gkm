# Unverified frontier brief: vc33 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed API surface, then reproduce any useful frontier observations before probing level 7. I’ll keep frame analysis symbolic and only commit a composed-leg solution after clone verification.
- Baseline reproduced: the solver reliably reaches level 6 in 148 moves, so level 7 is the sole target. I’ll now use the existing probe’s documented clone/setup pattern, adapting only its symbolic summaries for the new frame.
- The reproduced level-7 state is a three-column chamber system with three unequal marker pairs (colors 11, 14, and 15), eight small cyan pads, and three closed vertical gates. The coarse lattice correctly found no effects because the pads are only 2×2; I’m tightening the probe to their exact visible centers and testing the existing multi-reservoir relay leg, whose contract is the closest structural match.
- Exact-pad experiments identify the controlled object: each cyan pad shifts the adjacent fluid boundary, carrying embedded marker tips vertically. A single upper-right outward pad reduces the total same-color height mismatch from 74 to 72, giving a concrete dense metric. I’m testing whether repeated strictly improving visible-control choices solve directly; if it stalls, I’ll add bounded lookahead that permits neutral staging moves.
- Strict greedy progress reaches a real bottleneck (mismatch 44), confirming that neutral staging is required to open or traverse the pressure gates. The bounded lookahead probe is still running; it is capped at 1,200 symbolic states per stage and searches only visible pads/active gates.
- The existing single-platform gate leg does not advance here within its verified search envelope, while its alignment finisher reproduces only the same premature bottleneck. This confirms level 7 is a coupled relay: three marked platforms must exchange pressure, and optimizing one pair alone destroys the staging needed by the others. I’m switching the dense metric to horizontal gate traversal across all three pairs, with height used only after traversal.
- The first traversal is now reproduced in only three actions: stage the two upper reservoirs, then click the newly active upper gate; the color-15 platform crosses one chamber and horizontal mismatch drops sharply. The coupled multi-stage search is continuing under a 3,000-state cap to determine the safe traversal order for the remaining platforms.
- The coupling order is now clear and reproducible: cross color 15 through the upper gate, align it vertically, then three lower transfers lock that pair (its markers disappear as a completed subgoal) while staging the next platform. I’m encoding that observation as a general lexicographic score—unresolved pairs, uncrossed chambers, then post-crossing height mismatch—and verifying a bounded visible-control search against the whole level.

## Preserved local probes

- `probe_level4.py` (13644 bytes)
- `probe_level5.py` (4716 bytes)
- `probe_level6.py` (3765 bytes)
- `probe_level7.py` (7566 bytes)
- `probe_level7_score.py` (3289 bytes)
- `probe_level7_search.py` (5641 bytes)
- `reproduce_level4.py` (5779 bytes)

Run or inspect the smallest relevant probe before writing another one.
