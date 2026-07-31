# Unverified frontier brief: ar25 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The reproduced dense scores give two strong placements: mode 1 uniquely peaks at right×5, and the small J-piece uniquely achieves full target coverage at down×4, left×15. Mode 3 has three tied top-overlap placements, so I’m validating its top-ranked states directly with the required fourth `USE` and a full downward scanner sweep.
- Those top placements do not complete the level, so target overlap alone is not sufficient for the layered third field. I’m widening only the ambiguous dimension: keeping the independently verified mode-1 peak, testing the four plausible J-piece silhouette placements, and enumerating each distinct bounded mode-3 state once. This is about 1,200 candidate configurations and uses cloned prefixes to avoid redundant replay.
- The bounded exhaustive check rules out all 1,156 configurations under that assumption. Before expanding the positional product, I’m checking the remaining action properly: action 6 was inert only in the initial scanner context, so I’ll test it in every selection mode and after staged alignments. A context-dependent confirm/transform action would change the solution structure entirely.
- Action 6 is not inert after selection: it changes each of the three movable modes, and the failed diagnostic stopped only because of a missing local helper import. This is likely the missing rotate/transform mechanic. I’m now comparing repeated action-6 states by compact component signatures and tile-level changes before doing any more positional search.
- Action 6 only changes sub-tile highlighting; repeated uses leave every logical tile and component unchanged, so it is not a transform. A key earlier-level clue is that the “scanner” is itself a positioned layer, not merely a validator. I’m ranking the horizontal mode independently now; if it has a unique target row, fixing it up front removes the 22-step sweep from every candidate and makes a broader positional search practical.
- The horizontal layer has a unique occlusion minimum at down×9, a strong dense-progress signal consistent with earlier levels. I’m fixing that row and the J-piece’s unique full-coverage placement, then exploring all 21 vertical-layer positions against the complete 289-state third-mode graph. The third mode is traversed as a BFS per prefix, so this costs roughly 24,000 cloned steps rather than replaying every path.
- The search is still running cleanly; it has not emitted a winning prefix yet. The state cap is bounded, but full-frame keys retain some sub-tile selection animation, so this pass is taking longer than the logical 21×289 estimate. If it misses, I’ll quotient keys to 3×3 tile majorities before widening any other dimension.
- That complete product slice is negative: with the J-piece at its unique maximum, no vertical-layer position and no third-mode position works at horizontal down×9. I’m transposing the test now—fixing the strongest vertical and J placements, then checking every horizontal-layer row against all third-mode states. This distinguishes a bad scanner-row inference from a bad J-target inference without broadening both at once.
- The transposed slice is still executing within its 21 bounded prefixes. I’m preserving the exact reward test on clones rather than relying on the overlap heuristic; no live moves are being committed during this search.
- The transposed slice is also negative, so neither scanner row nor vertical placement alone explains the miss; the small piece’s highest raw overlap is the faulty assumption. Its board shows two symmetric composite silhouettes, and the right-hand bottom placement (down×4, left×10) loses visible overlap only where another layer occludes it. I’m testing that symmetric placement next across every vertical and third-mode state at the strongest scanner row.
- The symmetric J-placement slice is still bounded and running; no reward has appeared in the completed prefixes so far. I’ll use its result to decide whether to vary the J-piece vertically or revisit the horizontal dense metric.
- The right-bottom J placement is also ruled out across the full vertical/third-mode product. Two upper placements remain from the silhouette geometry (down×1 with left×10 or left×15); I’m testing both in one bounded pass at the scanner’s unique dense optimum.

## Preserved local probes

- `probe_bfs.py` (788 bytes)
- `probe_exhaustive.py` (1703 bytes)
- `probe_level6.py` (7819 bytes)

Run or inspect the smallest relevant probe before writing another one.
