# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The one-turn optimization is constrained by the courier’s fixed 21-turn loop, so I’m testing a structural alternative: fill middle-right before bottom-right. A left-held right-side cargo can pass up column 8 only while bottom-right is empty; reversing those two deliveries may eliminate the late courier dependency entirely. I’ll measure target fill and theft after each placement before investing in a full route.
- The reversed-delivery strategy is slower and lets color 15 reclaim a completed slot, so I’m discarding it. The shorter-contact state is genuinely carrying the cargo—the visual difference was courier timing, not failed pickup. I’m now completing the exact ten-action reward search from that state alone; this is the smallest remaining state space and can exploit its extra turn in a way the reused suffix could not.
- The short-state search has completed seven layers (5,430 transitions). It has not yet found reward, but the state count remains bounded; three layers remain. The key difference from the standard state is visible in the courier timing, so I’m continuing rather than pruning by the misleading “no empty outline” metric.
- Depth nine found a genuine all-eight-cargo state at turn 69 with suffix `[4,5,1,3,2,5,2,5,1]`. The final layer is now testing whether any turn-70 action can reward; if the cap preempts it—as before—the result will prove that this staging family is one action short, not merely that our hand suffix was poor.
- The tenth layer is expensive because it expands 4,394 distinct turn-69 frames. I’m keeping the hard 50,000-transition cap; it is nearly reached. The known all-filled turn-69 path has already been directly shown not to reward at turn 70, so this layer is only checking whether another hidden state does.
- The exhaustive staging search proves the current timing family cannot win. The expensive part is not cargo travel; it is the four-turn post-placement interception. I’m tracing color 15 during the second cargo’s shortest wall-gap route to see whether the avatar can meet and dismiss it en route, replacing interception moves with delivery moves rather than adding them.
- A more promising reuse of the courier mechanic is available: instead of spending the second remote block directly on bottom-middle, place it on the surviving courier’s original row-7 port. From the second pickup, staging at columns 4–8 costs no more than the direct delivery—and sometimes less—while potentially replacing the courier’s long trip to the far-left cargo. I’m probing those five ports with bounded dismissal searches and autonomous target traces.
- There is one integration detail I need to verify before further game search: the baseline reports exactly 600 moves even though `players.py` has no level-9 dispatcher. That suggests the tail of the level-8 leg may spill actions into level 9. I’ve been using the harness wrapper’s pristine level-9 handoff as instructed; now I’m checking the documented solve loop to ensure the final implementation will actually start from that same state.
- The courier-port experiment confirms the courier can consume the staged block, but it delivers bottom-middle and still leaves the same late middle-right cycle. I’m checking the preserved probe’s earlier cooperative route outcomes now; those routes stage and recover cargo around the courier rather than committing to the current direct-transfer family, and may contain a higher-progress turn-60 state we have not reproduced yet.
- The preserved cooperative probe is still replaying its many earlier branches and has not reached the filtered outcomes. I’ll give it one more short poll; if it remains unproductive, I’ll stop it and extract only the named action sequences into a compact verifier, as with the suffix search.
- The preserved cooperative branches do not outperform the direct route. I’m checking one last zero-cost structural variation before redesigning: reverse the two right-side cargo pickups while preserving the same turn-40 bottom-slot state. It follows a different corridor history and may put color 15 on the adjacent contact square one turn earlier even though the endpoint and action count are unchanged.
- Before changing strategy, I’m resolving the 600-move baseline anomaly with a read-only runtime probe. I’ll let the actual solve loop reach its level-9 dispatcher and record that entry frame. This verifies whether level 8 spills actions into level 9; if it does, the missing turn may be an integration bug rather than a level-9 route problem.

## Preserved local probes

- `probe9.py` (13826 bytes)
- `probe9_alt.py` (2466 bytes)
- `probe9_candidate.py` (3993 bytes)
- `probe9_courier.py` (1563 bytes)
- `probe9_current.py` (19683 bytes)
- `probe9_handoff.py` (968 bytes)
- `probe9_picksearch.py` (3443 bytes)
- `probe9_pickstate.py` (1320 bytes)
- `probe9_reroute.py` (2385 bytes)
- `probe9_reverse.py` (2306 bytes)
- `probe9_search.py` (13865 bytes)
- `probe9_stageports.py` (2512 bytes)
- `probe9_suffix.py` (2795 bytes)
- `probe9_thieftrace.py` (1222 bytes)
- `probe9_variants.py` (3719 bytes)
- `probe9_verify.py` (7540 bytes)
- `wip_glue_notes.md` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
