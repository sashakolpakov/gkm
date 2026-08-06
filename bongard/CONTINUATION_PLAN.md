# Bongard continuation plan

The objective is a usable headless visual proposer on the complete official
ShapeBongard corpus. The immediate work is to measure the new Python
visual-semantic path honestly. It is not to produce a SEALED number at any
cost.

## 1. Preserve A1 as a terminal failure

A1 is finished and failed. Its command receipt is
`sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`;
its terminal scoring-failure digest is
`sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`.
All 48 proposer calls succeeded: 37 proposals emitted an accepted soft claim,
10 were direct-only attrition, and 1 was rejected by the typed parser. All 37
scorer calls were transport errors. There were zero scores, labels remained
withheld, and no calibration, semantic accuracy, or negation evidence exists.

The consumed A1 seed and durable successor are respectively
`f9ee0fc4433df603049734153ae5eeac7e7227873fd2f3f36bc163449f107857`
and
`sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78`.
Retain them as history, not as permission to retry the selection.

## 2. Preserve the A2 source-mutation invalidation

A2 removes unsupported scorer-schema keywords while leaving exact cue and
witness validation in the Python decoder. That repair changes the protocol
digest, so A2 is a new experiment rather than an A1 retry. Its frozen
identities are:

- protocol: `sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca`;
- predecessor ledger: `sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78`;
- fresh no-reroll seed: `eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1`;
- durable successor: `sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8`.

A concurrent agent edited `bongard/typed_visual_proposal.py` after A2 froze its
protocol and cohort. The live grammar digest then differed from the frozen
protocol, and the process exited without writing a Stage-A terminal artifact.
The incident file digest is
`sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.

Process output showed 48 proposer and 34 scorer launches only. Their outputs
were lost and cannot be reconstructed as results. Labels were not revealed;
A2 provides no fitted calibration, accuracy, semantic inference, or negation
evidence. Its selected cohort remains consumed and may not be rerun.

For any later separately authorized Stage-A experiment, the scientific
question would remain narrow: did its frozen scorer family produce a valid
descriptive calibration artifact under its preregistered protocol?

A successful fit means only:

- soft claims were emitted often enough to observe;
- blind ordinal score records were valid;
- both fixed score bins met their cluster minimum;
- labels were joined after score commitments;
- the campaign and command receipt cold-replay exactly.

It does not mean “bird-like” is proven, the proposer is good, samples are
independent, or SEALED evaluation is authorized.

Neither A1 nor A2 fits. The metadata-only capacity audit is now complete. Under
the corrected HD constituent-disjoint policy, the remaining DRILL reservoir
has exact maximum 24 = 24 `bd` + 0 `hd`; 48 is impossible. The earlier 28-unit
upper bound applied within-new-batch HD disjointness but failed to seed the
constituent-token exclusion set from the complete A2 ledger. Every remaining
DRILL HD pair shares at least one constituent with that projected set. Any A3
must use the already frozen selector and one fresh no-reroll seed, and accept the
resulting seed-ranked greedy capacity without changing score-bin edges,
confidence, decision boundary, families, or polarity after inspection. Run it
only from an immutable committed worktree.

Launch that worktree through the production CLI with a fresh empty private
`PYTHONPYCACHEPREFIX` and `python -B`. The receipt binds `.py` source bytes, so
this is required to exclude stale or forged bytecode-cache execution. Do not
use the injectable Python API as the live authority; its alternate source-root,
transport, and verifier hooks exist for tests. Record the interpreter version,
interpreter bytes, complete command, and initially empty cache directory in
the external A3 precommit.

The source receipt hashes every potentially authoritative Bongard Python
module but explicitly excludes the exact non-authoritative
`bongard/semantic_checker.py` sidecar. This realizes the un-Lean invariant:
installing, editing, or deleting an optional checker cannot change the receipt
identity, while any authoritative Python-source change still invalidates the
run.

Before that seed exists, A3 changes `minimum_clusters_per_bin` from 12 to 8.
For two Bonferroni-adjusted bins at 90% confidence, eight gives Hoeffding radius
0.480161 and is the smallest count that can possibly place an interval wholly
above or below the fixed 0.5 boundary. Seven or fewer can never decide. This is
a prospective capacity repair, not a response to A3 scores; it makes A3 less
precise than A1/A2 and does not authorize Stage B.

The larger-corpus audit also separates two quantities that had been conflated:

| population | exact-unused after A2 |
|---|---:|
| train + validation | 10,069 / 10,200 |
| `ff` | 2,998 |
| `bd` | 3,456 |
| `hd` | 3,615 |

The 24-unit ceiling is caused by demanding constituent-disjointness across the
complete predecessor ledger and the new batch, not by lack of images. A fresh
seed changes the selected BD representatives and their order, not the maximum.
A3 is therefore BD-only and cannot support HD or mixed-family generalization.
A next-generation calibration frame should admit exact-unused training tasks
while excluding reserved DEV/SEALED
semantic keys, treat shared generators as dependence rather than silently as
independence, and blind-score both held-out panels of each task before opening
either label. Evaluation must remain strictly held out. The current HD
pair-level partition cannot support constituent-disjoint evaluation and must
eventually be rebuilt at the attribute level.

## 3. Stage B remains unauthorized

Neither the failed A1 receipt nor the invalidated A2 incident can authorize
Stage B. There is no current successful Stage-A receipt from which to build it.
If a separately authorized future receipt ever exists, Stage B must bind its
exact cache/launcher/exposure parents and use one fresh 256-bit seed generated
after those inputs and the Stage-B policy are frozen. No rerolls.

Before A2 exposure, exact-pair-only accounting reported:

| family | maximum disjoint tasks |
|---|---:|
| `bd` | 16 |
| `hd` | 8 |
| total | 24 |

That table was theoretically wrong: different HD pairs can reuse the same
constituent attribute. Recomputing against every task and panel exposure in the
full A2 successor gives 16 `bd`, 0 `hd`, total 16. Stage-B schema v2 archives
the excluded constituent inventory and recomputes it during execution and cold
replay. The default request for 24 fails closed before pixels, exposure, or a
model call. A 16-task BD-only run would remain a descriptive pilot and cannot
meet `minimum_selected_clusters = 24` or authorize SEALED.

Persist all selected exposures before starting the task executor. For every
task, cold-reconstruct one of the only accepted terminal forms:

- `complete` with a valid complete run archive;
- `support_rejected` with the exact reconstructed support gate and freeze;
- `proposal_error` only for a typed rejected proposal attempt.

Transport, preparation, gate-construction, persistence, or replay failures are
batch-fatal infrastructure errors. They are not class predictions.

Report over every task in any separately authorized complete
maximum-cardinality selection:

- proposal status counts;
- exact 12/12 support-gate coverage;
- both-query-correct rate among gated tasks;
- fully determinate, abstention, and error rates;
- `bd` and `hd` slices;
- the descriptive simultaneous intervals already fixed by policy.

Keep `inference_mode = descriptive-only-pending-family-stratified-power-audit/v1`
and `dependence_design_authorized = false` regardless of numerical passage.
Publish a Stage-B result only after strict cold replay; absence of a run is not
a pending numerical result.

## 4. Do not open SEALED

The visual-semantic official-test path is hard-disabled in both `cli.py` and
`benchmark.py`. A valid Stage A plus a descriptive DEV run is not sufficient
to remove that stop.

An authorizing design would need, at minimum:

1. a frozen exact-key population/frame digest;
2. a fresh auditable seed after frame and protocol freeze;
3. family-stratified sampling without replacement and a correct weighted
   estimator;
4. preregistered bins, sparse-bin behavior, thresholds, and stopping rule;
5. a defensible fixed-potential-outcome/no-interference assumption, or repeated
   executions that measure model stochasticity;
6. a power analysis that treats generator clusters, not panels, as units;
7. DEV-only model selection followed by one final SEALED report of actual
   performance.

Until that exists, SEALED remains inaccessible.

## 5. Fill the known perception gap

The direct catalog is small. It counts components, owned holes, endpoints,
branchpoints, cycles, certified X crossings, curvature reversals/runs, and
S-/U-like strokes. It does not directly represent:

- point contact versus overlap;
- exterior angular gaps and owner-labelled contact rays;
- oblique or parallel angle bands;
- inside/between/around relations;
- repeated-part correspondence;
- articulated object parts;
- open categories such as bird-like or tool-like.

Add a direct capability only when its extractor is panel-only, typed,
scenario-preserving, source-bound, uncertainty-aware, and cold-replayable.
Start with fixtures that include constructive positives, certified nonmatches,
ambiguous rasters, and extraction failures. A feature name is not an
implementation.

For open categories, improve the operational soft scorer instead of pretending
to formalize prose into truth. Keep the one-panel blind boundary, fixed ordinal
vocabulary, witness citations, family-level calibration, and four
dispositions. Measure proposal attrition and scorer calibration separately.

## 6. Diagnose reverse predicates directly

For every support rejection, archive and inspect:

- proposed direct atoms and soft cues;
- per-panel direct scenario outcomes;
- ordinal cue packets and calibrated intervals;
- positive-side and negative-side alignment separately;
- whether the exact complement would have fit, for diagnosis only;
- generator family and shared attribute/morphology cluster.

Do not promote or execute the complement. A complement win identifies wrong
orientation, spurious correlation, or a broken observation boundary. It is not
a solver success.

## 7. Keep Python authoritative

All new predicate execution, calibration, selection, replay, benchmark
decisions, and scientific artifact IDs must have complete Python semantics.
Lean or another proof checker may only consume an already-frozen artifact as a
detached optional sidecar. Deleting that sidecar must change no result,
decision, or ID. There is no planned Lean migration and no Lean dependency on
the benchmark path.

## Completion criterion

The current record is complete only as failure accounting: A1 failed and A2 was
invalidated without a terminal artifact. Before any new calibration is
declared, remaining clean capacity and source-dependency freezing must be
resolved. Stage B remains unauthorized. This phase is not complete merely
because the code builds or because a favorable complement can be found.
