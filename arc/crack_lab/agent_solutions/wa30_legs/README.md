# wa30 legs artifact

Latest replay-validated leg-library state promoted by `gkm_legs.py`.

<!-- BEGIN GENERATED: ARTIFACT_DETAILS -->
- Game: `wa30`
- Verified through level: 9
- Replay validated: True
- Final replay path length: 596
- Complete published ledger charge: 1458
- Complete published ledger: L1=112, L2=78, L3=95, L4=47, L5=405, L6=225, L7=145, L8=204, L9=147
- Current operational checkpoint charge: 1243
- Current operational checkpoint records: L2=30, L4=87, L5=405, L6=225, L7=145, L8=204, L9=147

The complete ledger and clean-source hashes are in `arc/manuscript/artifact_history/wa30/manifest.json`. The artifact root and `wip_context` retain the final clean replay state and the original dirty continuation evidence, respectively.
The `marginal_C` field is positive net retained-description growth per source file; additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARTIFACT_DETAILS -->

Files here are the clean state to resume from. New runs seed the scratch
workspace from this directory before asking a proposer for the next level.
