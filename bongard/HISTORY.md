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
- pure-Python predicate, IR, evidence, evaluation, selection, persistence,
  decision, ID, and cold-replay semantics;
- a detached, non-authoritative optional proof-checker sidecar whose presence,
  failure, disagreement, change, or deletion cannot change Python authority;
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

That 24 was the exact pre-A3 capacity. Exact v3 replay against A3's successor
ledger later certified zero strict DRILL capacity: no eligible task and no
eligible generator group remain under the same policy. The certificate digest
is `sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1`.
DEV remains 16 BD + 0 HD under the successor ledger.

The same audit found that this strict number had been misreported as corpus
exhaustion. The full non-test split has 10,200 tasks, of which 10,069 exact task
IDs remain absent from the A2 ledger (FF 2,998; BD 3,456; HD 3,615). The small
capacity is an independence-policy consequence. It reveals a calibration
design problem: fitting a scorer and testing semantic transfer need different
frames. A future calibration design should use the larger exact-unused
training population with explicit shared-generator dependence, score both
held-out panels before label reveal, and reserve a separately constructed
attribute-level HD evaluation split.

## A3 terminal scientific failure

A3 was then launched from the frozen source frame with a headless Codex
proposer and scorer. It completed every attempted transport and exited 2 as a
canonical scientific failure, not an operational crash. Its exact failure
reason was `calibration score bins are underpopulated: 1`.

- command receipt:
  `sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681`;
- terminal failure:
  `sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb`;
- proposer calls: 22 successful, comprising 15 accepted soft claims, 6
  direct-only records, and 1 typed-parser rejection;
- scorer calls: 15 successful out of 15 attempts;
- ordinal scores: eight `0`, one `0.5`, and six `1`;
- lower bin `[0, 0.75)`: 9 clusters, 1 affirmative;
- upper bin `[0.75, 1]`: 6 clusters, 5 affirmatives;
- fitted calibration and Stage B: none.

The fixed minimum was eight clusters in each bin. With only 15 scoreable
claims, meeting both minima was mathematically impossible. Intended-bin
orientation was 13/15 and the exact complement was 2/15; the naive
`score >= 0.5` orientation was 12/15 and its complement was 3/15. Negation did
not win. A3 measured a recruitment/bin-power failure.

The parser rejection was also traced precisely: the forbidden-code expression
matched the prefix `def` in the ordinary word `defines`. The parser was fixed
after A3 to require a complete forbidden-keyword match. The fix cannot revise
the recorded outcome or make the exposed tasks reusable.

The post-run executable audit found that A3's launcher digest
`134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477`
authenticated the JavaScript wrapper, not the native client it spawned. The
receipt also recorded `codex-cli 0.146.0`; it did not commit the native bytes.
The currently installed native digest is
`sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02`,
but that post-hoc value is not causal evidence about A3. This is an
authentication limitation, not evidence that the client actually changed.

A3 exposed 22 tasks and left 10,047 exact-unused train/validation IDs: FF
2,998, BD 3,434, and HD 3,615. Complete-release authentication hashed
official-test bytes, but no official-test task or panel was selected, exposed
to the proposer or scorer, evaluated, or scored.

Forensic replay also showed exactly what A3 omitted: all 264 panel
descriptions were audit-only, only 15 one-panel soft bundles were scored, and
nine deterministic atom instances were never evaluated as formulas. The 36
accepted cues produced 17 supported, 4 ambiguous, and 15 unsupported
judgments, but their citations contained no part, style, facing-axis,
curvature-orientation, or gestalt record. The cited low-level records therefore
authenticated context; they did not entail the semantic claims.

The run also made the representation hole explicit. Rich descriptions of the
twelve panels were recorded but used only for audit. The proposer made one
irreversible tiny-catalog-plus-bundled-soft-claim guess; synthesis merely
lowered that guess, and the scorer collapsed one to four cues by minimum into
`0`, `0.5`, or `1`. There was no description-to-atomic-facts-to-candidate
search. That diagnosis led to a Python-authoritative design using one-cue soft
atoms, an atom-by-panel score matrix, deterministic positive
atom/conjunction selection without `Not` or polarity flipping, structured
grounding, and label-blind recruitment powered for attrition and bin occupancy.
Python remains the sole runtime semantic authority. The target Lean boundary is
a detached, process-isolated, non-authoritative sidecar over persisted immutable
Python artifacts. That runtime fact does not yet complete the un-Lean migration: the
later attempt-three provenance still binds an earlier whole-tree commit
containing checker source, its config retains the legacy
`python-canonical/lean-optional/v1` literal, and the optional checker API remains
an in-process audit utility pending process isolation.

## Atomic operational successor

The description/matrix/selection successor is now executable. It freezes 12
isolated neutral vision descriptions, proposes 1--12 single-phrase atoms from
the labelled descriptions alone, records every atom/support-panel observation,
and chooses a deterministic positive conjunction of at most four atoms. Query
sources are unavailable until that formula is frozen. Two query descriptions
and two selected-atom observations complete the exact 29-call success path;
predictions cross a durable boundary before labels are materialized.

This implementation also corrects the theoretical evidence error found during
adversarial review. `operational_nonmatch` has its own persisted atomic record
and can reject a panel only inside an explicitly operational archive. In the
general truth lattice it is `indeterminate`, not `certified_absent`. The
archive permanently denies calibration, semantic-truth, benchmark, and
official-test authority. Calibrated semantic selection is hard-disabled until
Python can cold-validate a real typed calibration artifact and interval rule.

The first production precommit authenticated all 12,000 tasks and the exact
A3 successor ledger before metadata-only selection. Its original N=1 frame
historically contained ten exact-unseen training tasks from already exposed
Basic-shape generators, digest
`sha256:3246017440379de1e49f695503536f75062626d2de36bdab9112e96281e269a8`.
The selected exact-task exposure was durably persisted before any selected
panel was hashed. This frame was intentionally useful only for an exploratory
transport/synthesis smoke; it is not independent evaluation.

Before the first live atomic run, the offline genuine-receipt harness completed
all 29 calls, predicted both synthetic queries correctly, and cold-replayed
without a model. That is a protocol test, not a Bongard result. No live atomic
model call or new panel exposure had occurred at that milestone.

The first live N=1 attempt was then launched from commit
`62ea577f5d86d109577f4f5e49b8b4866eb76c92`, tagged
`bongard-atomic-pre-smoke-20260806`. A setup invocation first rejected a cache
store at mode `0755`; atomic stores require `0700`. That invocation persisted
no exposure and consumed nothing. After permissions were corrected, the
command persisted cache, config, and exact-task exposure, consuming the
selected task, but persisted neither prediction nor terminal. It will not be
rerolled.

Forensics matched reason digest
`2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`
exactly to `failed run precommit is not canonical JSON`. The runner had been
entered and returned a typed `AtomicSmokeRun`. Fallback terminal construction
attempted a JSON clone of its frozen `MappingProxy` precommit and failed.
Normal terminal construction contains the same deterministic defect, but the
outer error does not establish which exception first entered the fallback
path. The underlying status, phase, output, and successful model-call count
were not recoverable. The honest successful-call count is unknown in `0..29`.
Without a prediction artifact, labels could not be materialized or revealed.
This is an operational failure, not a Bongard score, and supports no
calibration, semantic, benchmark, or official-test claim. The sanitized
incident record is
[`data/atomic_smoke_n1_operational_failure_v1.json`](data/atomic_smoke_n1_operational_failure_v1.json).
Its file SHA-256 is
`2cf35e733c9a392999ec904660b2b0bf17814c253e3936476023f3e815fc14ad`.

Atomic attempt two then ran exactly once from commit
`d0864525146a05795c030674fa0159feb43913c1`, tagged
`bongard-atomic-successor-pre-smoke-20260806`. Its active predecessor was the
first attempt's successor
`sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a`,
and its historical input universe contained nine IDs, digest
`sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed`.
It durably appended exposure successor
`sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d`.

The journal closed with exactly 13 intents and 13 validated results: twelve
neutral support descriptions and one text-only atom proposal. The proposal
receipt and schema were valid. All ten emitted observer questions ended in the
question mark demanded by the prompt, but the shared soft-cue parser rejected
U+003F. The exact error was `invalid positive_description: soft cue
positive_description contains a forbidden prose character U+003F`, phase
`atom-proposal`, reason digest
`34b41a10ae89287ed97c875c6833047ff5896a7081debd144f484833292fe42f`.

No support-scoring call, formula, selection archive, query call, prediction,
label materialization, label reveal, or score occurred. The run, journal
terminal, and command terminal persisted, and cold replay passed. This is an
implementation-contract failure, not evidence about vision, predicates,
negation, or Bongard performance. The selected task is consumed. The sanitized
record is
[`data/atomic_smoke_attempt2_proposal_contract_failure_v1.json`](data/atomic_smoke_attempt2_proposal_contract_failure_v1.json),
file SHA-256
`242ebc5914020a683a6f34a0b50688bf3190f4c4cbd6d345d15ebb5e775eb6b3`.

## Atomic attempt three: terminal no-exact-separator failure

Attempt three ran exactly once from commit
`89b7124da62a5dd62a55abe7025bbaec8f90794b`, annotated tag
`bongard-atomic-attempt3-pre-smoke-20260806`. It was the distinct successor of
`sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d`
and selected from the exact eight-task universe with digest
`sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9`.
The fixed non-Bongard preflight and seed-independent exclusive claim both
passed. The claim protects only the canonical predecessor path; the journal
separately forbids resume or retry.

The journal closed 25 intents and 25 validated results: twelve isolated support
descriptions, one text-only proposal, and twelve support-scoring calls. The
sole question was `Is a small triangle attached to a tilted quadrilateral?`.
Every positive support was `present`. The negative outcomes were three
`operational_nonmatch`, two `present`, and one `indeterminate`.

The run terminated honestly at `support-scoring` with
`NoExactSeparatorError`: `no atom is total and present on every positive
support panel;
diagnostic_digest=b0d204d9eac2f36a66f13790dcb0eefe19f4b4ed99058e61fccd365c0e3dff14`.
The indeterminate observation made the only atom non-total. Resolving it to
nonmatch would not repair the two negative `present` observations. No formula
or selection archive was frozen; no query source was read or scored; no
prediction was persisted; query labels were neither materialized nor revealed;
and there was no score. The exact closed 25-call prefix cold-replayed.

The post-hoc affirmative-support orientation was 9/12 versus 2/12 for its
complement, counting indeterminate incorrect. This is explicitly not an
accuracy, score, calibration, semantic, benchmark, or negation claim. Negation
did not win the diagnostic. Complete-release authentication hashed
official-test bytes, but no official-test task or panel was selected, exposed
to a proposer or scorer, evaluated, or scored.

This failure exposed the next theoretical hole. The phrase satisfied the
surface grammar but bundled two shapes, relative size, orientation, and
directed attachment. The frozen descriptions retained relational information;
the proposer did not factor it, and the scorer lost object-role and
size-direction distinctions. The next representation must provide
candidate-independent typed scene graphs with stable object IDs and factorized
shape, closure, area, orientation, and owner-labelled contact relations for
Python to evaluate. The complete two-loop point-contact signature remains
unimplemented.

The selected task is consumed and may not be rerolled. Its exposure successor
is
`sha256:66678615dd766dcababfd57cb0435dfc2e18a366bd2a806127afb00a5b1ecfe6`.
Seven tasks remain in that historical frame, digest
`sha256:5dc36a2336abdb46d8096f0951739c3d825fb6e88833cb9b19735b426d1df357`.
The sanitized machine record is
[`data/atomic_smoke_attempt3_no_exact_separator_v1.json`](data/atomic_smoke_attempt3_no_exact_separator_v1.json),
file SHA-256
`533d57f8e0757ecd819c4e5bd95eeb8f2d2478193a366ba67178cf291ee953fc`.

The A2 incident also led to source-bound v2 command receipts and durable
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
