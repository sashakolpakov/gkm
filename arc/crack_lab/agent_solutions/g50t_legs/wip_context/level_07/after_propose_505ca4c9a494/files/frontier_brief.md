# Unverified frontier brief: g50t level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then the frontier brief if present, and keep all experiments bounded and symbolic.
- The brief points to a likely hidden-history cooperative gate puzzle, but those notes are unverified. I’ll now reproduce the baseline, then inspect only the indexed harness/API ranges and the smallest preserved probes needed to validate level 7.
- Baseline is reproduced at level 6. The preserved full probe mixes useful observations with several expensive searches, so I’m separating the compact observational portion first: avatar/action deltas, reachable positions, special surfaces, and a few concrete USE histories.
- The observations are now reproduced: actions 2/3 move the color-9 avatar by one grid cell from the start, action 5 is inert there, the color-14 mover advances on successful avatar movement, and the avatar has only 22 reachable positions with one special contact. The special commit `[2,2,3,5]` changes the barrier phase but does not itself grow reachability; repeated identical commits only oscillate. I’m now testing concrete hidden histories rather than merging visually identical states.
- A concrete timing effect is now verified: reaching the same switch after two reversible movement loops, then pumping the autonomous mover, increases avatar reachability from 22 to 25 and moves the autonomous block across the barrier. That gives a real dense-progress signal. I’m locating the earliest growth point and the newly exposed special frontier so the reusable leg can chain this behavior rather than hard-code level coordinates.
- The new reachability is temporary: pressing either exposed surface resets the visible maze to the 22-position start, so the missing unit is a two-part cycle—time the autonomous crossing, commit the newly reachable remote surface, then reopen the crossing to reveal the commit’s hidden effect. I’m testing that cycle directly now.
- The remote commit does not immediately restore avatar reachability, but it changes the autonomous mover’s later route: after another reset cycle the mover appears above the first barrier while avatar reach remains closed. I’m extending that post-commit run and tracking both mover position and barrier components; reachability alone is no longer sufficient for this intermediate stage.
- The second synchronization is now clear: after the remote commit, a timed bottom-switch press followed by ten patrol steps exposes the remote surface again (reachability 22 → 24). I’m turning this into a repeated observational cycle that searches only short timing prefixes, commits the newly exposed surface, and rechecks the true reward after every stage.

## Preserved local probes

- `probe_chain.py` (1835 bytes)
- `probe_gate2.py` (2441 bytes)
- `probe_hidden.py` (2839 bytes)
- `probe_l7.py` (7147 bytes)
- `probe_latch.py` (1867 bytes)
- `probe_long.py` (2085 bytes)
- `probe_macro.py` (1875 bytes)
- `probe_repeat.py` (2511 bytes)
- `probe_retime.py` (1638 bytes)
- `probe_search.py` (817 bytes)
- `probe_stage2.py` (1744 bytes)

Run or inspect the smallest relevant probe before writing another one.
