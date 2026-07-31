# wa30 legs artifact

Latest replay-validated leg-library state promoted by `gkm_legs.py`.

<!-- BEGIN GENERATED: ARTIFACT_DETAILS -->
- Game: `wa30`
- Verified through level: 9
- Replay validated: True
- Final replay path length: 597
- Complete published ledger charge: 318
- Complete published ledger: L1=43, L2=20, L3=32, L4=50, L5=39, L6=23, L7=28, L8=34, L9=49
- Current operational checkpoint charge: 318
- Current operational checkpoint records: L1=43, L2=20, L3=32, L4=50, L5=39, L6=23, L7=28, L8=34, L9=49

The complete ledger and clean-source hashes are in `arc/manuscript/artifact_history/wa30/manifest.json`. During active acquisition, the artifact root and `wip_context` retain the clean replay state and restart evidence. Publication authority comes from the compact schema-v2 release and its frozen receipt; terminal cleanup may prune mutable WIP after all release consumers are bound.
The `marginal_C` field is positive net retained-description growth per source file; additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARTIFACT_DETAILS -->

Files here are the clean state to resume from. New runs seed the scratch
workspace from this directory before asking a proposer for the next level.
