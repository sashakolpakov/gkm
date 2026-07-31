# Unverified frontier brief: lf52 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/perception surface. I’ll begin with the indexed summaries, then reproduce any prior observations with compact clone probes before changing the solver.
- Baseline reproduces four completed levels. The preserved probe contains unverified hard-coded paths, so I’ll use it only as a shape reference; I won’t trust or import its prior path artifact. Next I’m extracting the documented harness entry point and making a fresh compact level-5 observation from the current solver.
- The fresh no-checkpoint replay is still solving the earlier levels, which confirms why the harness normally resumes from its validated checkpoint. I’m letting this bounded observation finish once; after that I’ll reuse the live level-5 setup only through the documented run surface and keep individual probes short.
- Fresh probes confirm the avatar-like controllable object is the bordered upper carrier assembly: left/right move it, while up/down are context-dependent and initially only advance the turn marker. Coordinate action 6 selects and transfers pegs. The dense measure is now explicit: legal bridge transfers preserve peg count; captures reduce it from two to one, but the remaining peg must still reach a specific winning slot. I’m reproducing the preserved decision-point sequences to map that endgame.
- The one-peg state is not terminal because that peg is still inside a vertical carrier. A second bordered bridge carrier moves independently enough to create transfer alignments; plain key-only search finds none because coordinate transfers are still required. I’m now probing only four lattice-distance destinations per visible peg and treating successful transfers as macro edges, which keeps the search symbolic and bounded.
- The first macro frontier is slower than the desired probe cadence because it retains deep clone chains. I’m capping this run and, if it does not return immediately, will switch to replay-from-root paths with a smaller prioritized frontier so clone cost stays predictable.
- The bounded search confirms there is no short arbitrary transfer, but the geometry reveals a targeted cooperation pattern: park the empty bridge carrier in a fixed board slot so it becomes pinned, then return the peg carrier independently. I’m testing that specific alignment—three up, one right, three down—rather than expanding the broad frontier.
- A direct click probe shows the lone peg is no longer selectable—it is enclosed in the filled carrier—so the endgame is navigation, not another peg jump. The remaining dense subgoal is docking that filled carrier into a board slot. I’m mapping the wall/rail components and the carrier’s constrained motion to identify the docking route.
- The carrier remains unselectable at every reachable vertical stop, including the lower rail endpoint; coordinate clicks only affect the turn/cursor overlay. That rules out hidden click interaction. I’m running a fast raw-key navigation search now, excluding the overlay row so the frontier measures only physical configurations.

## Preserved local probes

- `probe_fresh_l5.py` (2038 bytes)
- `probe_l5.py` (3536 bytes)
- `probe_l5_endgame.py` (3166 bytes)
- `probe_l5_key_bfs.py` (1081 bytes)
- `probe_l5_macro_bfs.py` (3365 bytes)
- `probe_level2.py` (6693 bytes)

Run or inspect the smallest relevant probe before writing another one.
