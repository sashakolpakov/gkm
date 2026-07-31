# Unverified frontier brief: sk48 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 5 at 364 actions.
- Remaining real-action budget under the harness cap: 236 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/perception surface. I’ll begin with `solver_index.md`, then `frontier_brief.md` if present, reproduce any needed observations, and keep probes compact before changing `legs.py` and `players.py`.
- The preserved brief confirms the tight constraint: level 6 begins after 364 real actions, leaving 236. Its behavioral notes are unverified, so I’m treating only that harness budget as a benchmark and will reproduce the level-6 mechanics from a pristine run.
- The pristine probe identifies a two-carriage layout with two independent edge-mounted tethers and a bottom reference: the left carriage is paired with three `9` blocks, the top carriage with three `8` blocks. Actions `1–4` currently move/extend the selected left carriage; coordinate action `6` changes which carriage is selected. I’m now isolating selection, orientation-relative controls, collision/attachment, and whether either carriage can hand blocks across the other’s reach.
- The earlier macro paths do reproduce useful partial progress: all three `8`s can be staged as a vertical train at rows 10/16/22, and all three `9`s as a horizontal train at columns 14/20/26. Crucially, exact geometry alone does not trigger reward; the probes show blocks can be geometrically aligned yet detached from the intended tether. I’m testing the remaining attachment/ownership condition rather than optimizing those coordinates blindly.

## Preserved local probes

- `probe_l6.py` (2049 bytes)
- `probe_l6_interleave.py` (5942 bytes)
- `probe_l6_ownership.py` (5034 bytes)
- `probe_l6_paths.py` (7983 bytes)
- `probe_level2.py` (1043 bytes)
- `probe_level2_sequences.py` (2189 bytes)
- `search_l6.py` (2122 bytes)
- `search_l6_coop.py` (3283 bytes)
- `search_l6_finish.py` (4211 bytes)

Run or inspect the smallest relevant probe before writing another one.
