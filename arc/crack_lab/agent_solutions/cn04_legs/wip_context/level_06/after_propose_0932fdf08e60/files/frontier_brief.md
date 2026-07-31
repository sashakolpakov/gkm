# Unverified frontier brief: cn04 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the documented harness/perception surface. I’ll begin from the index, reproduce any preserved observations I rely on, then probe level 6 symbolically and keep the player as leg composition only.
- The baseline run is still starting because matplotlib is building a one-time font cache. Once it returns, I’ll isolate level 6 on clones and avoid broad frame dumps.
- Baseline is verified at five levels. I’m now making a small level-6 observer that replays the existing players, then reports only color counts, components, and one-step clone deltas—including coordinate selection candidates inferred from object centroids.
- The level is a packing/matching layout on a 3×3-cell lattice: three selectable centers begin in one staging area, while four black silhouettes lie elsewhere. I’m extracting each selection’s normalized footprint and each target’s footprint, then I’ll match shapes under quarter-turn rotation and plan collision-free translations.
- The colored cluster is one selected polyomino; the black components are other selectable polyominoes, not targets. I’m reproducing the verified level-5 solution at action granularity to identify the actual completion event and a reliable dense measure before searching level 6.
- The verified prior solution confirms the reusable mechanic: each piece is oriented, then pushed upward and rightward until blocked so it interlocks with the growing assembly. I’m validating that macro against level 5, then searching only the small rotation combinations for level 6; translations are derived by “push until blocked,” not brute-forced step by step.
- The first level-6 rotation sweep found no completion, so the new configuration changes more than orientation. I’m checking the macro itself against level 5 and measuring where level-6 pieces stop; that will separate an incorrect movement abstraction from a new ordering/direction requirement.
- The prior level’s four-turn “full rotation” is mechanically meaningful—it is not equivalent to zero actions—so I’ve expanded the bounded search to include that state. The validation/search is still running; no broad state dump is being produced.
- The prior frame gives a clean dense measure: immediately before reward, every already-placed black piece is one connected component; the final right move joins the last selected piece and triggers completion. I’m now using “number of occupied connected components” as the progress metric and doing bounded single-piece searches that must reduce it before moving on.
- The first bounded connectivity search is still evaluating clone states. It is constrained to one selected piece, depth 32, and 5,000 observational states; if that frontier is insufficient I’ll tighten the state key to the selected footprint rather than expanding the budget blindly.
- The dense search is working: it reduced six components to three with short, verified paths before I stopped an expensive Python flood-fill. I’m replacing only that measurement with an equivalent vectorized connected-component labeler, then resuming the same bounded search.

## Preserved local probes

- `probe_connect.py` (2577 bytes)
- `probe_l6.py` (2302 bytes)
- `probe_prior.py` (1539 bytes)
- `probe_search.py` (2359 bytes)

Run or inspect the smallest relevant probe before writing another one.
