# ls20 legs artifact

Current stable artifact from the most recent successful `ls20` leg-library crack.

- Game: `ls20`
- Scope: levels 1 through 4
- Result: replay-validated L4 run
- Run log: `run.log`
- Solver files: `legs.py`, `players.py`, `solve.py`
- Debrief: `legs_log.md`

Final run summary from `run.log`:

```text
level 1: reached=1 marginal_C=55 total_C=55 F=0.100
level 2: reached=2 marginal_C=18 total_C=73 F=-0.540
level 3: reached=3 marginal_C=2 total_C=75 F=-1.500
level 4: reached=4 marginal_C=2 total_C=77 F=-2.460

=== ls20: reached level 4 | validated=True | total_marginal_C=77 | F=-2.460 ===
```

`checkpoint.json` is included as captured workspace state, but the final validated
run log is the authoritative result for this artifact.
