# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll begin from `solver_index.md`, then use `frontier_brief.md` if present. I’ll keep discovery clone-only until the level-7 mechanic and a dense progress signal are reproduced, then make the smallest reusable leg/player change and validate with `gkm_try.py`.
- Baseline is reproduced exactly: the solver enters level 7 with `levels=6`, `moves=196`, and a valid replay. I’m now narrowing in on the preserved room-transition probe and the existing gravity/support leg, because the previous frontier’s key claim still needs independent reproduction before it can drive implementation.
- The preserved claims now separate into two reproducible signals: - toggling the left-edge control twice advances its vertical marker by one band without killing the avatar; - after the eight-action staged opener, a left move advances the camera by two bands, while gravity/right continuations die. That gives a concrete dense measure—cumulative band shift—but the second-room action set still needs to be mapped compactly. I’m checking the existing staged and bounded-search probes next, using only their symbolic summaries.
- The action semantics are now independently confirmed: `3/4` move the color-9 avatar, clicking a thin color-12 support expands it, supports remain interactable at a two-row vertical offset, and the x=3 color-8 edge control flips gravity from any sampled y. I’m exposing those two level-specific affordances as optional parameters on the existing gravity-room search (defaults unchanged), then testing the preserved 300-state continuation before deciding whether a new leg is needed.
- The 300-state continuation is still running after a minute, confirming the prior warning: the old macro search’s replay reconstruction is the bottleneck, not the frontier size. I’m stopping this pass and replacing only its clone-search internals with compact retained states or a room-local search, while preserving the same verified action set and progress measure.
- A compact room-local BFS found the next safe transition: move left, expand the lower adjacent support, flip gravity, cross left, then flip back. It produces a verified +2-band shift with the avatar alive. The search is correct but too slow for runtime use, so I’m now extracting the repeating room macro from successive symbolic states rather than embedding BFS in the final player.
- The next frame is a mirrored version of the earlier chamber, but the previously expanded lower-middle support persists. A 300-state atomic BFS found no new down-gravity shift, which suggests the needed transition is a staged gravity macro rather than another local step. I’m enumerating only those physically justified macros—nearby support, flip, short crossing, flip back—so the probe stays small and interpretable.
- The blocked section is now explained and verified: the apparent obstacle is a gravity-dependent hazard, so walking/clicking it cannot cross. The successful cooperative affordance is to stage the thin support one row below the destination, flip upward, cross over the hazard, then flip back onto that support. This yields an +8-band safe shift and preserves the avatar—exactly the kind of remote-support dependency the level is testing.
- The repeated-cycle probe hit the harness’s workspace-taint checker with an internal `Path` lookup error before the game started; no game state was touched. Since earlier probes passed, I’m isolating the triggering local probe syntax/file and will keep the experiment within the same documented harness surface.

## Preserved local probes

- `probe_level7.py` (5105 bytes)
- `probe_level7_bfs.py` (3806 bytes)
- `probe_level7_controls.py` (1908 bytes)
- `probe_level7_macro.py` (2995 bytes)
- `probe_level7_macros.py` (3165 bytes)
- `probe_level7_progress.py` (5879 bytes)
- `probe_level7_room2.py` (1455 bytes)
- `probe_level7_search.py` (1024 bytes)
- `probe_level7_stages.py` (5000 bytes)

Run or inspect the smallest relevant probe before writing another one.
