# Unverified frontier brief: bp35 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The third ascent reaches the far-left wall, but the remaining gravity phase then has no horizontal escape. That isolates the missing cooperative subgoal: before consuming the second control pair, a support must change the inverted landing so the avatar can reset rightward instead of drifting permanently to column 0. I’m testing only the five visible supports in that exact inverted context.
- A genuine cooperative transition is now verified: expanding either lower-left support before the second gravity pair changes the landing from column 2 back to column 1 without death. That is the first support branch that changes a meaningful state rather than just pixels. I’m following that landing to see whether it exposes the next controls or the goal corridor.
- No single stage-1 support changes the third inverted landing; that branch is conclusively a left-wall dead end. I’m stepping back one chamber and checking for a support-assisted rightward reset while four controls are still available—the point where a zigzag can preserve enough gravity resources to continue upward.
- The useful macro is now quantified: expanding support `(4,1)` before the pair trades a one-band ascent for retaining two later controls. All lower support choices preserve position but gain no height. I’m applying that verified “short supported ascent” once, then enumerating the next support landing to continue the staircase.
- The two-control short-ascent branch also exhausts its remaining switches, so support placement alone is not enough. The remaining finite ambiguity is control choice: the same first ascent can preserve different pairs such as `{27,57}` or `{27,51}`. I’m now searching only complete gravity-pair macros over those observable control sets, reconstructing every branch from one shallow state and retaining cumulative ascent as the dense score.
- The macro diagnostic found the modeling error: it paired gravity flips but omitted the normal-gravity move where ascent actually occurs. I’m correcting each successor to `optional normal move → chosen flip → crossing move → chosen flip`, and scoring cumulative one-step camera rise across the replay so repeated-looking chambers still retain their true progress.
- The corrected macro search is now distinguishing real repeated-room progress: it has found a two-pair branch with cumulative ascent 6, versus the earlier plateau at 4. That branch currently ends with no controls, while neighboring states retain one; I’m letting the deduplicated frontier compare those continuations before adding support toggles.
- The tail macro frontier is a closed trap even when its switch coordinates vary. The level-6 mechanic suggests a more promising earlier cooperation point: level 7 has a thin support directly under the opening route, which must be staged only after the avatar has occupied or passed its cell. I’m verifying whether clicking that occluded support persists once the avatar steps away.
- Staging the opening row-6 support is real, but it blocks the established ascent and every immediate gravity pair stays at zero height, so it is not the bridge by itself. I’m checking the remaining opening supports one at a time through the verified long prefix; this isolates which remote expansion survives to the late chamber without derailing the ascent.
- Opening support `(2,2)` produces the strongest verified alternate: after the prefix and one recovery flip, the avatar is upright at column 4 with three controls still visible, instead of the column-2 four-control trap. I’m using that as the new macro-search root; it has both rightward reach and enough gravity resources for another ascent.
- The staged `(2,2)` branch is a genuine breakthrough: the verified macro frontier rises from dense score 0 to 6, 11, 16, then 19 while staying alive—far beyond the old trap’s ceiling. The current best has exhausted visible controls but remains at column 4; neighboring slightly lower states retain controls, so I’m allowing the bounded search to finish its last useful comparisons.
- The score-19 branch is another high-looking but control-less chamber, so it is not sufficient. I’m adding exactly one nearby support toggle to each complete gravity macro from the staged `(2,2)` root. This preserves the verified macro structure while allowing the missing landing cooperation, without reopening arbitrary click search.

## Preserved local probes

- `probe_level7.py` (5105 bytes)
- `probe_level7_alt.py` (2444 bytes)
- `probe_level7_alt_support.py` (2389 bytes)
- `probe_level7_altfinal.py` (1867 bytes)
- `probe_level7_bestfirst.py` (4108 bytes)
- `probe_level7_bfs.py` (3806 bytes)
- `probe_level7_clean.py` (1976 bytes)
- `probe_level7_collapse_trace.py` (3370 bytes)
- `probe_level7_continue.py` (11576 bytes)
- `probe_level7_controls.py` (1908 bytes)
- `probe_level7_cycle.py` (8142 bytes)
- `probe_level7_endgame_options.py` (8835 bytes)
- `probe_level7_first_supports.py` (3677 bytes)
- `probe_level7_fresh_search.py` (3766 bytes)
- `probe_level7_hazard.py` (2588 bytes)
- `probe_level7_initial_combos.py` (2584 bytes)
- `probe_level7_macro.py` (2995 bytes)
- `probe_level7_macro_search2.py` (5577 bytes)
- `probe_level7_macros.py` (3165 bytes)
- `probe_level7_observe.py` (2524 bytes)
- `probe_level7_occluded.py` (2100 bytes)
- `probe_level7_opening_pairs.py` (2525 bytes)
- `probe_level7_progress.py` (5879 bytes)
- `probe_level7_room2.py` (1455 bytes)
- `probe_level7_search.py` (1024 bytes)
- `probe_level7_stage_extra.py` (4625 bytes)
- `probe_level7_stages.py` (5000 bytes)
- `probe_level7_start_all.py` (2156 bytes)
- `probe_level7_switches.py` (2368 bytes)
- `probe_level7_tail.py` (5609 bytes)
- `probe_level7_tail_search.py` (3505 bytes)
- `probe_level7_transitions.py` (3150 bytes)
- `probe_level7_two_macros.py` (6007 bytes)

Run or inspect the smallest relevant probe before writing another one.
