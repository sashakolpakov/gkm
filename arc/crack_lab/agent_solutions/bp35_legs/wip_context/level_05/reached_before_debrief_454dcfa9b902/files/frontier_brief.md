# Unverified frontier brief: bp35 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Both dense-progress transitions are now reproduced: the color-12 sequence gains five bands, and the aligned color-14 removal gains another five bands into a chamber where the prize is visible. I’m making the minimal reusable change: let the gravity-room leg observe the eighth column and treat only nearby color-12 cells as contextual toggles, then level 5 will compose that leg.
- The patched full replay is running. Earlier levels still use the same leg behavior; the only broadened observations are the additional on-screen column and a local color-12 interaction, so this test will directly show whether the shared search can now traverse level 5 within its existing bounds.
- The first integrated attempt preserved levels 1–4 but found no level-5 route within the old 900-state breadth-first budget. The mechanics are correct; the search order is the remaining issue. I’m switching this reusable search to prioritize verified camera ascent—the dense progress signal already provided by `band_shift`—so it explores the two five-band transitions before spending states on lateral/no-op variants.
- The ascent-prioritized replay is still searching, with the same 900-state cap. If it succeeds, I’ll immediately replay-validate the produced path; if it does not, I’ll instrument only the search’s compact best-height/depth frontier to identify whether the cap or a missing contextual action is responsible.
- The prioritized search still stops before a solution, so height alone is insufficient under the old cap. I’m using the preserved final-chamber probe now: it deterministically replays only the already verified subgoals, then performs a capped 2,500-state, depth-20 search with all eight columns visible. This isolates the missing final interaction from the expensive full-route search.
- The final-chamber probe is still in the four-level setup replay. Once it reaches the isolated chamber, its output will enumerate every one-action effect symbolically before searching, which should distinguish a missing affordance from a search-order problem.
- The setup is slower than the isolated search itself, but it remains within the preserved bounded probe. I’m letting this single run finish because it avoids another full-route hypothesis cycle and directly tests the final chamber.
- The probe has passed the usual setup duration without output, likely because the newly prioritized level-4 search changed replay cost. I’ll allow one final short interval; if it does not return, I’ll stop it and temporarily isolate the already known level-4 behavior from the experimental level-5 search ordering.
- The isolated output reached the final chamber before interruption and revealed why its broad probe was slow: it clicked every distant special cell. The chamber itself is simpler—avatar at column 4, a passage at the eighth column, stacked supports above it, and the prize at column 3. I’m replacing that broad enumeration with a tiny corridor probe that tests only rightward passage, aligned supports, and the prize approach.
- The corridor probe is in setup replay. Its frontier work is deterministic and tiny: three right moves, then only aligned color-14/15 interactions, followed by left moves toward the visible prize. This should yield an exact causal trace rather than another state-space sweep.
- The setup again dominates runtime because levels 2–4 plan on clones each run; the actual level-5 probe has not emitted yet. I’ll keep this single deterministic run going, since it is the narrowest remaining experiment.
- The corridor trace reproduced two more mechanics precisely: entering column 7 gains two bands, and removing the first aligned support gains one. Removing the second support immediately kills the avatar because the color-15 hazard has descended into the shaft. I’m testing the required ordering now: transform that hazard while it is two rows above, then remove the second support and approach the prize.

## Preserved local probes

- `probe_level5.py` (1382 bytes)
- `probe_level5_after12.py` (3043 bytes)
- `probe_level5_contexts.py` (2781 bytes)
- `probe_level5_corridor.py` (2112 bytes)
- `probe_level5_deeper.py` (2917 bytes)
- `probe_level5_final.py` (2753 bytes)
- `probe_level5_map.py` (2909 bytes)
- `probe_level5_search.py` (1035 bytes)
- `probe_level5_unknown.py` (2295 bytes)

Run or inspect the smallest relevant probe before writing another one.
