# Unverified frontier brief: su15 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed summaries, then reproduce any needed level-5 facts through compact clone probes. After identifying the mechanic, I’ll keep `players.py` as composition-only and verify the full run with `gkm_try.py`.
- Baseline is clean and reproducible: the current solver completes four levels in 83 moves, so level 5 is the active frontier. I’m now extracting only the harness entry points and existing leg interfaces needed to create a level-5 clone and summarize its objects/actions.
- The first probe identifies a reused mechanic in a harder configuration: four playfield `6` squares must merge pairwise (`6+6→15`, then `15+15→11`) and the final square must enter the `9` ring. Two color-7 agents move on every click; clicking one can hold it while the other advances, so collision avoidance now has two independently reachable bodies. I’m first testing whether the existing general merge-and-deliver leg already handles this configuration before adding any new leg.
- The direct reuse test is computationally heavier than the initial probe because the old state key collapses both moving agents into one centroid. I’m letting this bounded attempt finish once; if it fails or remains inefficient, I’ll refine only that state representation and candidate steering in a minimal reusable leg.
- The two-step merge is reproducible in either order, and it preserves all tracked mass while the two cutters follow distinct trajectories. The old leg fails because it (a) treats unrelated color-10 singles as merge targets and (b) keys both cutters by one averaged center. I’m minimally generalizing that existing leg with a selectable starting stage and a per-cutter positional state, then level 5 can reuse it with the `6`-square cohort.
- The generalized leg now solves the new configuration: the end-to-end harness reaches level 5 in 98 moves with replay validation passing. I’m doing a final determinism and syntax check now; the successful run also confirms the inferred reward condition, since the level increments only after the merged `11` cohort is delivered to the ring.
- Implemented level 5 successfully. - Reused and generalized the moving-cutter merge leg to track multiple agents and select the relevant merge cohort: [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/legs.py:189) - Added composition-only level-5 dispatch: [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/players.py:22) - Preserved compact level-5 probe scripts for reproducibility. Validation passed consistently: ```text RESULT levels=5 moves=98 replay_ok=True err=None ``` Syntax checks passed, and two consecutive full runs produced identical results.

## Preserved local probes

- `probe_dots.py` (903 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)

Run or inspect the smallest relevant probe before writing another one.
