# Unverified frontier brief: sp80 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with `solver_index.md`, then inspect only the documented ranges and observation/action surface needed to reproduce facts on fresh clones.
- The frontier notes are only hypotheses, so I’m treating them as an index. I’ll first establish the real baseline with the harness, then rerun the smallest preserved level-5 probe rather than trusting prior observations.
- The preserved probes launch the arena by reaching into a parent path, which is outside the stated clean-room surface, so I won’t execute them directly. I’ll reproduce their compact measurements through `gkm_try.py` by temporarily adding a level-5 diagnostic player that only uses the supplied `env` and `perception.py`; then I’ll replace it with the actual leg composition.
- Fresh evidence confirms four independently movable pieces. Selection is contextual action 6; arrows translate the selected piece by 3 pixels, and action 5 is the commit check. Three fixed sockets are above the chamber and one is on its left, so I’m now extracting exact piece/socket silhouettes and their admissible movement lattice before searching configurations.
- The current harness resumes from a validated checkpoint at level 4, so earlier-level diagnostics were correctly skipped. To reproduce the projection rule cleanly, I’ll temporarily rename that workspace checkpoint (without reading it), run the same harness from level 1, and restore the checkpoint immediately afterward.
- The reproduced earlier levels reveal the dense objective: interval connectivity. A piece wins by bridging separated socket projections, and redundant pieces can move without breaking the win. On level 5, the starting top projection is one 3-pixel step short between the left socket’s bar and the central network, so I’m testing the minimal one-step bridges first, while preserving overlap with the new side socket.
- A stronger dense measure fits every verified earlier sensitivity: cover both prongs of every fixed socket, not merely connect the socket components. Level 5 starts with exactly two uncovered prongs—the far-left top prong and the upper prong of the side socket—so I’m testing configurations that cover those while leaving the other six covered.
- The hand-built high-score layouts still miss a hidden constraint, so I’m switching to a bounded clone search over only dense-progress states: configurations whose movable projections continuously cover the full top target span and the side target span. I’ll cap it at 800 commits, use deterministic sampling, and print only the first verified plan.
- The bounded search is still running after the harness startup; it has not produced a result yet. I’m letting this capped batch finish, then I’ll either replay its verified plan or tighten the projection filter from the compact failure count.
- The dense-span hypothesis is now falsified by 800 independently replayed layouts. I’m going back to the smallest known instance—level 1—to identify what the fixed U-shapes actually encode, using the same symbolic blob/mask probe and a fresh harness start; this should distinguish projection matching from connectivity before spending more clone budget.

## Preserved local probes

- `probe_known_sensitivity.py` (2093 bytes)
- `probe_level5.py` (5404 bytes)
- `probe_level5_hypotheses.py` (2004 bytes)
- `probe_projection_rule.py` (1827 bytes)
- `probe_win_sets.py` (1949 bytes)

Run or inspect the smallest relevant probe before writing another one.
