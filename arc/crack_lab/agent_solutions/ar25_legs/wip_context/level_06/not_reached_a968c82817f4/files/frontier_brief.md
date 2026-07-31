# Unverified frontier brief: ar25 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the current workspace and use only the documented harness/perception surface. I’ll begin with the index, then reproduce any inherited observations before relying on them, and keep probes bounded and symbolic.
- The baseline run is still advancing through the existing five players. Once it reaches the level-6 frame, I’ll characterize objects and one-step action effects from clones without dumping the grid.
- The level-6 player slot is absent as expected. I’m adding a temporary observational probe that enters level 6 through the existing solver, then reports only component summaries and cloned one-step deltas for actions 1–7.
- The first probe identifies the controlled scanner: action 2 moves the full-width scan line downward; action 1 is blocked at the top. Action 5 switches control context, after which horizontal actions move a piece. I’m now isolating which colored piece is selected and its step size, then I’ll use a dense alignment score against the fixed silhouettes.
- The pieces are asymmetric in mobility: the color-4 shape has 14 legal horizontal states, while color 5 moves in two dimensions. Their strongest observed silhouette overlaps occur near `right ×5` and `down ×4, left ×10`. I’m validating that concrete staged configuration with the scanner before widening the search.
- The overlap hypothesis was not sufficient: the color fields reveal and occlude multiple disconnected layers, so raw overlap alone is misleading. I’m switching to a small observational BFS over the five effective actions, quotienting away the animated border; the first pass is capped at 3,000 states and depth 35.
- The dense score now gives a crisp first placement: the two-color horizontal layer peaks uniquely at `right ×5`. The smaller color-5 piece has four bbox-exact candidate silhouettes (`down ×1/4`, `left ×10/15`); none of those four completed with the first placement, so I’m checking all 14 legal horizontal states against those four placements and a full scanner sweep—56 bounded candidates.
- The failed 56-grid exposes a bad assumption: action 5 may cycle through more than two disconnected pieces before returning to the scanner. I’m testing repeated uses directly and identifying each selected entity from its cloned movement delta; this should reveal the true subgoal count.
- There are exactly four modes: scanner → horizontal two-color piece → small color-5 piece → a third multicolor piece → scanner. That explains the failed validation. I’m now finding the third piece’s best silhouette alignment, then I’ll validate all three placements with the required fourth use before sweeping.
- The third-piece top-overlap candidates still do not validate, so I’m removing that ranking assumption. I’ll exhaust its 244 observed legal placements against the four bbox-exact second-piece placements, keeping the uniquely strongest first-piece state and performing the correct fourth-use scanner sweep.

## Preserved local probes

- `probe_bfs.py` (788 bytes)
- `probe_exhaustive.py` (1703 bytes)
- `probe_level6.py` (7819 bytes)

Run or inspect the smallest relevant probe before writing another one.
