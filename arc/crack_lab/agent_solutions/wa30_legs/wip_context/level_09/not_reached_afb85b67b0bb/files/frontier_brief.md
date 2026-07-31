# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The frontier brief contains hypotheses only, so I’m treating none as established. I’m now running the required baseline harness; after that I’ll inspect only the small relevant definitions and reproduce the level-9 handoff and mechanics independently.
- The integration anomaly is now reproduced: level 8 advances and then executes nine further actions in level 9, leaving the avatar at cell `(9,15)` and one of eight target slots already filled. That is the real dispatcher entry, so I’ll solve from this exact state rather than a pristine level clone. Next I’m compactly mapping objects and testing all five actions on clones.
- The spill is now precisely reproduced: the validated checkpoint enters level 9 at `(8,8)`, then contributes exactly 13 actions `(U,D,D,R×10)`. During those turns a helper courier fills the first target slot; the avatar merely ends at the right edge. I’m testing pickup/contact from that real state now, using filled-slot count as the dense progress signal.
- The real timing creates a much better opportunity than the pristine probes: after two remote deliveries are routed, the thief and avatar meet on row 9 while the avatar is still carrying the second block. I’m testing whether that collision can dismiss the thief in-line; if so, it prevents the stolen-slot recovery detour and should fit comfortably under turn 70.
- The inline dismissal is verified and preserves the stolen slot, but a direct finish remains one cargo short: the surviving courier contributes only one more delivery before the cap. I’m now reproducing the preserved courier-port experiment because staging the dropped second block on its return lane is the one general mechanic that can turn one courier trip into two without adding a long hand-carry.
- The column-6 port is materially better: after dismissal, the courier is on a second target delivery by turn 70 instead of completing only one trip. I’m testing the complementary route now—leave the staged remote block for the courier and hand-deliver a nearby local block into bottom-middle. This composes the existing “feed a courier port” and hand-carry mechanics and is the first route with enough physical cargo in motion to fill all eight slots.
- The column-6 route now reduces the level to a one-turn timing problem: six slots are stable, the courier finalizes top-right, and reaches middle-right exactly when the cap fires; the staged block can fill bottom-middle much earlier. I’m enumerating the equally short thief-dismissal variants because they produce different courier phases—one alternative phase only needs to advance that final courier drop by one turn.
- Staging both remote blocks is the breakthrough: by turn 44 the thief is gone, both staged blocks have been consumed, and the avatar is already two cells from the remaining local cargo. That recreates the preserved near-win state one move earlier and with the courier at the target. I’m replaying the compact prior recovery suffix from this verified state now.
- A second reproduced constraint changes the integration problem: the checkpoint already consumes the harness’s 600-action global cap. The last 13 actions are level-9 spill, so even a correct level-9 route cannot execute through the real harness. I’m now measuring the documented checkpoint’s per-level costs and will remove only solver-side waste—especially cross-level spill—before integrating the verified level-9 leg.
- The 600-action cap is now directly reproduced, so a level-9 player alone cannot run. The only viable in-scope integration is a joint optimization of the level-8 tail and level 9. I’m inspecting the pristine level-8 handoff through the same documented wrapper, looking specifically for a route under roughly 72 turns so the verified level-9 plan has room.
- I’m switching from hand-route speculation to a bounded clone minimization of the proven level-8 leg. This preserves the verified action order and deletes only chunks whose replay still earns the level sooner, using actual reward—not frame-change—as the acceptance test. It is a compact way to expose no-op travel and over-waiting under the joint cap.
- A much shorter level-8 interception is visible from the raw timing: if the avatar goes through the nearby upper wall gap first, it meets the upper competitor on turn 6 instead of chasing it until turn 37. I’m verifying the reversed interception order now; this can recover a large fraction of the missing global budget while preserving more target cargo.

## Preserved local probes

- `probe8_candidates.py` (4200 bytes)
- `probe8_entry.py` (1367 bytes)
- `probe8_minimize.py` (2312 bytes)
- `probe8_trace.py` (2423 bytes)
- `probe9.py` (13826 bytes)
- `probe9_actual_candidates.py` (4474 bytes)
- `probe9_actual_ports.py` (8832 bytes)
- `probe9_alt.py` (2466 bytes)
- `probe9_candidate.py` (3993 bytes)
- `probe9_context.py` (1760 bytes)
- `probe9_courier.py` (1563 bytes)
- `probe9_current.py` (19683 bytes)
- `probe9_entry.py` (2521 bytes)
- `probe9_handoff.py` (1261 bytes)
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
- `probe_solver_timing.py` (980 bytes)
- `wip_glue_notes.md` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
