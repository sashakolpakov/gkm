# Unverified frontier brief: vc33 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with `solver_index.md`, then reproduce only the observations needed through bounded clone probes. I’ll keep level orchestration in `players.py` and mechanics in reusable legs.
- Baseline is confirmed: the current solver reliably reaches level 2 in 10 moves. There’s no preserved frontier brief, so I’m now inspecting only the indexed solver/perception ranges and the local harness entry points needed to create a level-3 clone.
- The level-3 clone exposes only coordinate action 6. It contains eight visible controls and three distinct moving marker colors (11, 14, 15); single clicks shift structured regions rather than acting as ordinary movement. I’m reducing each clone state to marker coordinates and cavity geometry next, then I’ll search against a dense alignment measure instead of raw pixel change.
- The mechanic is now verified on a clone: the controls transfer two-row “height units” between adjacent reservoirs, and the three same-color marker pairs are the targets. A 23-click relay—first draining intermediate reservoirs leftward, then transferring from the tall right reservoir—aligns all three pairs and raises the clone to level 3. I’m encoding that as a state-driven reusable transfer leg, not as a level-specific click list.
- The implemented solver now reaches level 3 in 33 total moves, and the harness replay validates successfully. I’m removing the temporary probe and doing one clean repeat to confirm the result comes solely from `players.py` and `legs.py`.
- Implemented level 3: - Added reusable reservoir-transfer leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/legs.py:83). - Added level composition in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/players.py:13). - Removed temporary probes. Verified twice: ```text RESULT levels=3 moves=33 replay_ok=True err=None ``` Python compilation also passes.
