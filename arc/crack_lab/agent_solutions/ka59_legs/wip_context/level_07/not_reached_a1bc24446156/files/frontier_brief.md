# Unverified frontier brief: ka59 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then any `frontier_brief.md`, reproduce observations with compact bounded probes, and keep level-specific orchestration out of `players.py`.
- The frontier brief records a failed search caused first by measuring the wrong color, then a corrected but still inconclusive target-directed search. I’ll treat all of that as hypotheses. First I’m checking the real baseline; next I’ll inspect only the small preserved level-7 probes needed to reproduce geometry and action behavior.
- The reproduced frame confirms three goal outlines and three movable rings/pieces: two selectable color-14 pieces and one large color-11 piece isolated in the upper-left chamber. The two color-13 objects cycle with period six on every arrow action. The preserved “best” path seats both small pieces but does not finish, so the dense progress metric must include transferring the large color-11 ring to the lower-left target. I’m now narrowing the contact/transfer mechanic rather than expanding another raw-frame search.
- A replay of level 6 reproduced the key transfer behavior: the selected small ring first pushes the large ring laterally, then an interlocked command moves the large ring across an otherwise sealed barrier. That means level 7 is an earlier mechanic in a new layout, not a new action. I’m searching only for the first verified displacement of the large ring—separately from each selectable small ring—with a compact object/phase state key.
- The first contact search is still running after 30 seconds, so it is larger than the simple routes tested previously. It remains bounded at 12,000 symbolic states per ring; if neither reaches the large piece, I’ll split the route at the observed transfer corridors instead of increasing that bound.
- No result has emerged at the bound yet. I’m stopping this run rather than spending more clone budget, then I’ll reduce the search state to the selected ring position plus the six-phase corridor state and search one ring at a time through explicit waypoints.
- The horizontal-ring contact search has also passed 60 seconds without finding a displacement. I’ll let this fixed 12,000-state bound complete once; that result will distinguish “wrong ring/path” from a search artifact.
- The cooperative transfer is now reproduced directly: a vertical ring underneath the horizontal ring pushes it from rows 51–53 to rows 33–35, across the sealed row-36 barrier. This explains why either ring alone cannot reach the large piece. I’m extending that same aligned push through the upper barrier and recording only ring/large displacements.
- The reverse relay works: selecting the horizontal ring beneath the aligned vertical ring and pressing up twice transfers the vertical ring from rows 42–47 to rows 27–32. That vertical piece now fits the three-cell gap in the upper barrier. I’m routing it through that gap, over the top of the left wall, and down onto the large ring—the exact approach needed to push the large ring toward its target.
- The upper-wall route is blocked exactly as the geometry predicts: the transferred vertical ring fits the barrier gap, but its six-cell height cannot use the three-cell corridor above the left wall. I’m now testing the level-6 pattern at that wall—align beside the large ring, push it laterally, and see whether the small ring is handed into the sealed chamber.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)
- `probe_l6.py` (5807 bytes)
- `probe_l6_final.py` (956 bytes)
- `probe_l7.py` (2431 bytes)
- `probe_l7_contact_bfs.py` (1275 bytes)
- `probe_l7_focus.py` (1670 bytes)
- `probe_l7_large_transfer.py` (986 bytes)
- `probe_l7_push.py` (988 bytes)
- `probe_l7_relay.py` (1953 bytes)
- `probe_l7_search.py` (4469 bytes)
- `probe_l7_select.py` (1549 bytes)
- `probe_l7_state.py` (2858 bytes)
- `probe_l7_trigger.py` (1329 bytes)
- `probe_transfer_trace.py` (1310 bytes)

Run or inspect the smallest relevant probe before writing another one.
