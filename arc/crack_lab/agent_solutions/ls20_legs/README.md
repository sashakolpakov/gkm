# ls20 legs artifact

Latest replay-validated leg-library state promoted by `gkm_legs.py`.

<!-- BEGIN GENERATED: ARTIFACT_DETAILS -->
- Game: `ls20`
- Verified through level: 7
- Replay validated: True
- Final replay path length: 365
- Complete published ledger charge: 760
- Complete published ledger: L1=40, L2=54, L3=86, L4=114, L5=138, L6=170, L7=158
- Current operational checkpoint charge: 760
- Current operational checkpoint records: L1=40, L2=54, L3=86, L4=114, L5=138, L6=170, L7=158

The complete ledger and clean-source hashes are in `arc/manuscript/artifact_history/ls20/manifest.json`. During active acquisition, the artifact root and `wip_context` retain the clean replay state and restart evidence. Publication authority comes from the compact schema-v2 release and its frozen receipt; terminal cleanup may prune mutable WIP after all release consumers are bound.
The `marginal_C` field is positive net retained-description growth per source file; additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARTIFACT_DETAILS -->

Files here are the clean state to resume from. New runs seed the scratch
workspace from this directory before asking a proposer for the next level.
