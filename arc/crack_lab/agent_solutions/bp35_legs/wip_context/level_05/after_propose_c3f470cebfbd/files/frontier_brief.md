# Unverified frontier brief: bp35 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin from the indexed surfaces only. I’ll inspect `solver_index.md`, then `frontier_brief.md` if present, reproduce any useful observations with bounded probes, and keep level-specific logic out of `players.py`.
- Baseline is reproduced: levels 1–4 complete in 113 moves, and level 5 has no player yet. I’m now checking the existing gravity-room leg and the preserved compact probe; every level-5 claim from the brief will be re-tested before use.
- The preserved probe is still replaying the expensive level-4 search before it reaches level 5; it has produced no level-5 evidence yet. I’m letting this bounded run finish once, then I’ll reuse a shallow level-5 root within each probe process so we don’t repeatedly pay that setup cost.
- The first reproduced observations confirm the raw action set is `(3, 4, 6)`: 3/4 move the color-9/11 avatar left/right, while bare 6 only advances the move counter. The context probe now checks the gravity toggle from every reachable lower-room column and tests each visible upper support as a staged interaction.
- The context test reproduced the safe transition: toggling from lower columns 0–3 terminates, while columns 4–5 survive and lift the avatar into the next chamber. Pre-clicking any upper support did not change that transition, so I’m narrowing the next probe to immediate, state-changing click affordances before and after the safe toggle, plus a finite test of the existing level-4 search.
- The harness setup remains the slow part; no new probe output has arrived yet. The current run is capped at 300 search states and depth 32, and it will report only clicks on observed special cells plus whether the finite reused search actually earns the level reward.
- A direct support experiment has identified the next meaningful subgoal: after the safe gravity lift, removing the support directly beneath the avatar changes the whole visible room, whereas removing the neighboring support only deletes that block. The new run is mapping that post-drop chamber and recording the proven level-4 action pattern for comparison.
- The post-drop map gives a clear route continuation: the avatar can move right into column 6, and only the support directly under that new position triggers another whole-room transition. I’m advancing that verified sequence one phase and mapping the next chamber; the dense progress signal is now “support removal causes room/camera transition while avatar survives,” not raw pixel change.
- The phase-3 probe is still in the deterministic setup replay. So far, the discovered route uses only existing gravity-room affordances—sideways movement, a global toggle, and removing the aligned support—so I’m aiming to generalize the existing leg rather than introduce unrelated level-specific actions.

## Preserved local probes

- `probe_level5.py` (1382 bytes)
- `probe_level5_contexts.py` (2781 bytes)

Run or inspect the smallest relevant probe before writing another one.
