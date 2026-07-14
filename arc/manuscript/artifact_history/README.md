# Manuscript artifact-history sidecar

This directory is a compact, read-only audit index for the ARC-AGI-3 manuscript.
It does not participate in cracking, promotion, workspace seeding, or WIP restart.
Those behaviors remain in `arc/crack_lab/gkm_legs.py` and continue to use the
artifact roots under `arc/crack_lab/agent_solutions/`.

For each game, `manifest.json` contains the complete published per-level ledger,
the final replay endpoint, public Git provenance, and hashes of the clean core files
retained at each available promotion event. The copied core is limited to the shared
leg library, level players, dispatcher, debrief log, and checkpoint files when they
were present. The complete dirty probe history remains, without pruning, in the
artifact's `wip_context/` tree. For example, `wa30` levels 1--3 were committed
together as one clean promotion event at `15c0a049`; this commit cadence does not
omit any level from the nine-entry ledger or from the final nine-level replay. Also,
`ls20` had its clean-through-L2 core preserved in the failed L3 attempt archive,
while the reconstructed manuscript sidecar has clean source evidence at every depth.

Rebuild and verify the sidecar from the repository root:

```bash
python arc/manuscript/build_artifact_history.py
python arc/manuscript/build_artifact_history.py --check
```
