# Unverified frontier brief: lf52 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then reproduce any prior observations from `frontier_brief.md` before relying on them, and keep probes compact and bounded.
- The frontier brief contains useful hypotheses but no trusted evidence. I’m now running the required baseline; after it reaches level 7, I’ll use the preserved small probe first and independently verify each mechanic through fresh clones.
- The clean baseline is confirmed at six completed levels. Replaying the preserved probe is taking time in the six established solvers, but it is the single clean setup pass; I’ll consolidate the level-7 experiments afterward instead of repeatedly paying that cost.
- The replay is still healthy but produces no intermediate output by design. I’m letting this one bounded process finish; once the level-7 callback runs, its compact component and four-action deltas should immediately confirm or falsify the brief’s claimed avatar and rail geometry.
- This from-scratch replay has exceeded the useful setup budget without reaching its callback. I’m stopping only that diagnostic process and switching to the validated local checkpoint path used by `gkm_try.py`; the level-7 observations will still be freshly reproduced from the documented environment surface.
- Fresh evidence confirms the brief’s first structural claims: the only key-controlled object is the color-12 carrier, and its rail graph has exactly 16 reachable positions. The initial peg and movable bridge are separated from the lower relay, so the existing visible-board solver correctly finds no move; this needs a new reusable wrapped-relay leg, not a level-6 drop-in. I’m now verifying the first transfer and how an occupied carrier is represented.
- The entire first relay is now independently reproduced: both objects load into the same bordered carrier, unload at distinct lower exits, then leapfrog over fixed and movable bridges. The viewport wraps exactly when the bridge reaches the right edge, and the post-wrap peg jump lands in the second chamber. I’m mapping that chamber’s carrier route next, with a concrete dense subgoal: place the empty carrier at `(54,22)` so the bridge can jump over the peg into it.
- The second-chamber clone search is the first heavier probe because rail motion can trigger viewport transitions. It remains capped at 300 observational states and depth 28; I’m waiting for this one precise target rather than broadening the search.
- The capped BFS is computationally inefficient because deep environment clones get progressively expensive, despite the tiny symbolic state cap. If this run does not return imminently, I’ll replace it with shallow path reconstruction and carrier-position keys—the same observational search, but without recursively nested clones.
- The carrier route is not a straight descent: its rail has gated turns at `(36,16)`, `(36,28)`, and `(42,28)`. I’ve verified each junction rather than assuming reversibility across wraps. I’m narrowing the remaining route from `(42,28)`; once the bridge is loaded, the chamber’s fixed-bridge geometry gives a dense, checkable relay sequence.

## Preserved local probes

- `probe_checkpoint_level7.py` (7358 bytes)
- `probe_level7.py` (988 bytes)

Run or inspect the smallest relevant probe before writing another one.
