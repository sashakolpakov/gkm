# ls20 legs artifact

Latest replay-validated leg-library state promoted by `gkm_legs.py`.

<!-- BEGIN GENERATED: ARTIFACT_DETAILS -->
- Game: `ls20`
- Verified through level: 7
- Replay validated: True
- Final replay path length: 393
- Complete published ledger charge: 362
- Complete published ledger: L1=43, L2=2, L3=45, L4=3, L5=72, L6=130, L7=67
- Current operational checkpoint charge: 362
- Current operational checkpoint records: L1=43, L2=2, L3=45, L4=3, L5=72, L6=130, L7=67

The complete ledger and clean-source hashes are in `arc/manuscript/artifact_history/ls20/manifest.json`. The artifact root and `wip_context` retain the final clean replay state and the original dirty continuation evidence, respectively.
The `marginal_C` field is positive net retained-description growth per source file; additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARTIFACT_DETAILS -->

Files here are the clean state to resume from. New runs seed the scratch
workspace from this directory before asking a proposer for the next level.
