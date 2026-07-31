# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- All one-step crossings are safe but produce no height, so the macro needs a longer traversal before the second flip. I’ll extend only the same staged templates to one–four crossing steps and zero–two settling steps, printing solely reward or live positive-gain outcomes.
- The support/gravity macro family is exhaustively negative in this chamber. The remaining documented local affordance is the adjacent color-15 hazard itself, which the existing hazard leg treats as shape-changing. I’ll now compare its single/repeated clicks before and after gravity flips, reporting only its shape, avatar survival, and camera gain.
- Hazard clicks are confirmed no-ops; only movement, support expansion, and visible gravity controls matter. I’ll now run a retained-state best-first search from the blocked chamber, prioritizing cumulative live camera rise while allowing the search to backtrack from dense-progress traps. It is capped at 600 expansions and depth 48, with nearby supports at the experimentally verified two-row reach.
- The best-first search has found a live +2 route after 48 expansions and is continuing to backtrack beyond it. Clone throughput is about one expansion per second here, so I’ll watch for the next progress jump; if it plateaus, I’ll stop before the full cap and extract the best route for a room-local continuation.
- The search’s dense measure increases regularly: +2 at path length 4, then +4 at length 8. That strongly indicates a four-action repeat rather than a combinatorial route. I’ve stopped the expensive run and will rerun only 100 expansions with exact progress paths printed, enough to identify and directly verify the cycle.
- The search reveals the repeat exactly: from the blocked chamber, `gravity flip → right → gravity flip → left` gains two bands and returns to the same avatar column; repeating it gains another two. I’ll directly repeat this dynamic-control cycle until reward, death, or the last visible control scrolls away.
- Two bare cycles exhaust the controls at only +4, confirming they are a tempting but insufficient loop. The missing cooperation must be a remote support staged before it scrolls. I’ll enumerate every currently thin support at four insertion points around one/two verified cycles, retaining only live routes that beat +4 or complete the level.
- A key untested affordance appears in the transition evidence: clicking an already expanded color-12 support contracts it back to area 5. The prior searches incorrectly treated supports as one-way expansion, excluding the very actions needed to reopen the next path. I’ll explicitly reproduce that toggle, then rerun best-first search with both thin and expanded nearby supports interactable.
- The support toggle is now independently confirmed: the same color-12 cell changes area `21 → 5` when clicked. The corrected search has a larger local branch set and is still exploring before its first height improvement; I’ll continue watching for a route rather than treating frame changes as progress.
- The corrected search still finds the bare +2 cycle first, but has not yet produced a higher route. It remains bounded at 300 expansions; I’m letting it backtrack through the newly available contraction states, which are precisely the branches absent from the earlier negative search.
- The search has reproduced +4 at length 8 and is now exploring contraction variants after that point. No higher live route has appeared yet; the process remains healthy and within the fixed 300-expansion cap.
- The corrected search has plateaued longer than useful at +4, so if it finishes without a jump I’ll use the verified toggle mechanic to enumerate contraction placements around the two known cycles directly; that is much cheaper than continuing broad state expansion.

## Preserved local probes

- `probe_level7.py` (5105 bytes)
- `probe_level7_bestfirst.py` (3911 bytes)
- `probe_level7_bfs.py` (3806 bytes)
- `probe_level7_clean.py` (1976 bytes)
- `probe_level7_controls.py` (1908 bytes)
- `probe_level7_cycle.py` (8142 bytes)
- `probe_level7_hazard.py` (2588 bytes)
- `probe_level7_macro.py` (2995 bytes)
- `probe_level7_macros.py` (3165 bytes)
- `probe_level7_progress.py` (5879 bytes)
- `probe_level7_room2.py` (1455 bytes)
- `probe_level7_search.py` (1024 bytes)
- `probe_level7_stages.py` (5000 bytes)
- `probe_level7_transitions.py` (3150 bytes)

Run or inspect the smallest relevant probe before writing another one.
