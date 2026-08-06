# Bongard history

This file keeps the results that matter after removing stale, contradictory
narratives. Machine-readable result JSON remains in the repository. The full
pre-rewrite documentation is recoverable from annotated tag
`pre-bongard-complete-rewrite-20260805`, which peels to commit
`1a71bb1560d41e4a908003a4efd6442e255602fc`.

## Falsified raster prototypes

The first fixed twelve-task development calibration tested four deterministic
raster feature groups. Every group selected margin `1e-9` and passed 0/12
strict support gates. The best group, moments/symmetry, classified 10/24
development query images and solved 2/12 puzzles. One border-clipped support
panel made a task unfittable; it stayed in the denominator as error.

Canonical record digest:
`sha256:cf02d58ab57fe1b44201c67d06f00faf06e77374b762c81ff5f61ef20aef93b6`.

The preregistered twelve-task headless PURE DRILL then produced:

- 0 completed episodes;
- 11 `support_rejected` episodes;
- 1 replayable `proposal_error`;
- 0 support-gate passes;
- 0 query releases and therefore no query-accuracy estimate.

Across the 132 executable support-panel outcomes, 46 were aligned, 10 were
reversed, and 76 were indeterminate. The aggregate is
`data/support_prototype_drill_result_v1.json`, digest
`sha256:38a89b3f78afa7c89f2f9dc881d209fce7b791ef3a346e54ee9ee3abaffa7fca`.

These runs established useful failure accounting. They did not establish a
working vision system or an official test score.

## Complete-corpus foundation

The repository moved from small public generator/gallery material to the
complete pinned ShapeBongard V2 release:

- 12,000 tasks and 168,000 panels;
- family counts `ff=3600`, `bd=4000`, `hd=4400`;
- primary split `train=9300`, `validation=900`, `test=1800`;
- complete PNG audit: RGB, 512x512, one frame, zero anomalies.

Exposure histories were reconstructed and converted to append-only,
content-addressed ledgers. Historical or semantically colliding tasks are not
treated as unseen merely because their exact task ID differs.

## Visual-semantic replacement

On 2026-08-06 the canonical design changed from prototype distances and broad
formal-language claims to:

- candidate-independent exact-byte visual witnesses;
- a finite positive direct catalog;
- at most one blind, ordinal, family-calibrated soft claim;
- four explicit evidence dispositions;
- conjunction evaluation inside retained correlated preprocessing scenarios;
- exact support freeze before query access;
- pure-Python evaluation, selection, persistence, IDs, and cold replay;
- an optional/removable proof-checker boundary;
- descriptive Stage A and Stage B only, with visual-semantic SEALED disabled.

The first visual-semantic Stage-A experiment, A1, used no-reroll seed
`f9ee0fc4433df603049734153ae5eeac7e7227873fd2f3f36bc163449f107857`.
Its exposure successor was durably committed as
`sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78`
before campaign pixel/model access. A1 then terminated as a scorer-schema
failure:

- command receipt: `sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`;
- terminal failure: `sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`;
- proposer calls: 48 successful, comprising 37 accepted soft claims, 10
  direct-only records, and 1 typed-parser rejection;
- scorer calls: 37 transport errors out of 37 attempts;
- successful scores: 0; labels remained withheld;
- fitted calibration, semantic accuracy, and negation evidence: none.

The frozen scorer schema used provider-incompatible `minItems`, `maxItems`, and
`uniqueItems`. Removing those keywords while retaining fail-closed Python
decoder checks changed the protocol identity. A2 was therefore a distinct new
experiment, not an A1 retry. It used protocol
`sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca`,
fresh seed
`eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1`,
and durable successor
`sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8`.

A concurrent agent edited `bongard/typed_visual_proposal.py` after A2 froze its
protocol and cohort. A2 was therefore invalidated by live source mutation and
exited without a Stage-A terminal artifact. The incident file digest is
`sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.
Process output showed 48 proposer and 34 scorer launches only; outputs were
lost, labels were not revealed, and no calibration, accuracy, or semantic
inference is valid. The same cohort may not be rerun. Stage B did not run and
is unauthorized by both A1 and A2.

The post-incident capacity audit exposed a second theoretical error. Exact HD
pair identity is not independence: two nominally different pairs can share a
constituent attribute. An intermediate 28-unit upper bound still failed to
project complete-A2 exposures into the HD constituent-token exclusion set.
With that projection, remaining DRILL has maximum 24 BD + 0 HD = 24, while DEV
against the complete A2 ledger has 16 BD and 0 HD. Stage-A and Stage-B schema
v2 now enforce this stronger relation.

The same audit found that this strict number had been misreported as corpus
exhaustion. The full non-test split has 10,200 tasks, of which 10,069 exact task
IDs remain absent from the A2 ledger (FF 2,998; BD 3,456; HD 3,615). The small
capacity is an independence-policy consequence. It reveals a calibration
design problem: fitting a scorer and testing semantic transfer need different
frames. A future calibration design should use the larger exact-unused
training population with explicit shared-generator dependence, score both
held-out panels before label reveal, and reserve a separately constructed
attribute-level HD evaluation split.

The incident also led to source-bound v2 command receipts and durable
operational failures for any post-precommit source change. Identity-preserving
canonical caches reduced the same synthetic Stage-A path from 161.15 s to
11.50 s and Stage B from 218.88 s to 51.10 s.

## Removed narratives

The cleanup removes obsolete prose reports and plans that mixed synthetic
renderers, Bongard-LOGO/action programs, categorical speculation, early crack
plans, and exploratory Phase-D notes with the current official-pixel track.
Their historical content remains at the annotated snapshot tag above.

No machine result JSON is removed. `/arc` is a read-only design reference and
is not part of this cleanup.
