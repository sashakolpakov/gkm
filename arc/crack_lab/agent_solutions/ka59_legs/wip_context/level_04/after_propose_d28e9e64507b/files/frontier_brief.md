# Unverified frontier brief: ka59 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the indexed harness/perception surfaces. I’ll first inspect `solver_index.md`, then `frontier_brief.md` if present, and reproduce any observation before relying on it.
- There is no prior frontier brief, so I’m starting a fresh clean lineage. Next I’m running the harness once to establish the current reached level and obtain the legitimate level-3 clone surface.
- The existing levels establish a consistent mechanic—coordinate-select a solid piece, then move it with arrow actions—but I’ll still re-verify level 3 rather than assume it transfers unchanged. I’m adding a compact probe that reaches level 3 through the existing players, then reports only color components and clone action deltas.
- The clone check exposed an important mismatch: clicking the hollow centers (and then visible edge coordinates) did not change which object moved. I’m treating coordinate semantics as unverified and testing a small set of swapped/edge points; this avoids building a plan on a false “selection” assumption.
- The level-1 control check reproduced the coordinate action correctly: it selects a second color-14 ring, confirming the existing leg. Level 3 is genuinely different: the two large color-11 rings are not selectable, so the small color-14 avatar must interact with them physically. I’m now probing contact from different sides and tracking ring positions as the dense progress signal.
- Contact tests show the new general mechanic: a small selected avatar pushes a large ring, and each successful push moves that ring by a 15-pixel “lane” step. Because the right ring begins flush against the wall, direct placement is impossible; the two rings likely must cooperate. I’m using a bounded clone BFS keyed only by the visible playfield (excluding the action-indicator row) to find and then replay-check the interaction sequence.
- The bounded search found a 33-move path and a fresh clone replay raised `levels_completed` from 2 to 3. It uses only the existing `move_steps` leg, so no new leg is justified; I’m adding the run-length-compressed composition to `play_level_3` and validating the complete solver through the harness.
- The full harness now reports `levels=3`, `moves=94`, `replay_ok=True`, with no error. I’m removing the two temporary probe scripts I created so the workspace retains only the solver change, then I’ll do one final replay validation.
- Implemented `play_level_3(env)` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_ka59_budgeted_medium_screen/players.py:62), composing only the existing `move_steps` leg. Final validation: ```text RESULT levels=3 moves=94 replay_ok=True err=None ```
