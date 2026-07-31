# Unverified frontier brief: su15 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll begin with the workspace index and frontier brief if present, then reproduce only the observations needed to reach and probe level 4. I’ll keep all inspection and experiments inside the stated clean-room surface.
- Baseline reproduced: `levels=3`, `moves=64`, replay valid. I’m now locating only the harness entry points needed to clone the level-4 state and the existing leg interfaces, then I’ll summarize level 4 symbolically rather than dump frames.
- The level-4 state is now reproduced (the harness reports `levels_completed=3`, meaning the fourth board is active). It exposes only coordinate action `6`. The initial object summary suggests a single ring and one small detached shape, but one playfield color is connected to the border, so I’m refining the segmentation with row runs and localized click deltas before inferring the mechanic.
- A direct ring-directed route is highly informative: the color-7 avatar follows a deterministic path, reducing the eight isolated color-10 cells to zero while approaching the ring. That gives a strong dense progress measure (`remaining color-10 cells`, then avatar-to-ring distance). I’m checking the zero-collectible transition separately because the clone raises an exception on the next naïve click, and I won’t encode a route until that end-state behavior is verified.
- The ring-entry test disproved “reach the ring alone”: the avatar can enter it while all eight color-10 cells remain and reward stays flat. Clicking a color-10 cell remotely removes it while the avatar also advances, so these cells are part of the objective/state, not ordinary walls. I’m now testing a staged route that clears them while deliberately preserving the lower-left cell for the final approach, avoiding the verified zero-cell failure state away from the ring.
- The bounded symbolic search is running over only observed coordinates: current color-10 cells, the ring center, and four local steering clicks. It is capped at 800 observational states and depth 14; invalid zero-cell branches are discarded rather than committed.
- The paired structure is now verified: the eight color-10 pixels are four linked pairs, and clicking either endpoint removes exactly that pair. This explains why treating them as eight independent collectibles misled the first search. I’m revising the dense measure to prioritize exact avatar–ring alignment while treating pair removal as an optional topology change, then running one deeper but still capped search.

## Preserved local probes

- `probe_dots.py` (903 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)

Run or inspect the smallest relevant probe before writing another one.
