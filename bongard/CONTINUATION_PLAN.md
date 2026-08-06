# Bongard continuation plan

The objective is a usable headless visual proposer on the complete official
ShapeBongard corpus. A3 has now measured the current Python visual-semantic
path honestly and failed its calibration fit. The description-to-predicate
search layer is now implemented as an exact operational atomic smoke. Its first
live N=1 attempt failed in the command wrapper after exact-task exposure; it
produced no Bongard result, and that task is consumed without reroll. The
wrapper defect is repaired and the successor is hardened with a non-resumable
per-call journal. The immediate work is to gate and run that distinct successor
once, report it honestly, then design richer typed perception and a defensible
evaluation frame. It is not to produce a SEALED number at any cost.

## 0. Preserve the first atomic N=1 operational failure

The live command used pre-smoke commit
`62ea577f5d86d109577f4f5e49b8b4866eb76c92` and annotated tag
`bongard-atomic-pre-smoke-20260806`. Cache, config, and the exact selected-task
exposure were durably persisted; no prediction or terminal artifact was
persisted. The selected task is consumed and must not be rerolled.

The CLI emitted `AtomicSmokeCommandError` with exact message `failed run
precommit is not canonical JSON` and reason digest
`2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`.
The runner was entered and returned a typed `AtomicSmokeRun`. Fallback terminal
construction then tried to JSON-clone the run's frozen `MappingProxy`
precommit. Normal terminal construction contains the same deterministic defect,
but the surviving error does not establish which earlier exception entered the
fallback path. The run's status, phase, output, and successful-call count are
irrecoverable. Record successful calls as unknown in the inclusive range
`0..29`, never as zero or 29.

Without a prediction artifact, labels could not be materialized or revealed.
The attempt has no score and authorizes no calibration, semantic, benchmark,
or official-test claim. The persisted content addresses are cache
`sha256:1094dfd6794d4dfd141b9d0d1c89cf648d5c7d57ea0a545868bc38df928f28a4`,
config
`sha256:9dad0a5f468d1e8f3c65f7b83ac1ce7d2072e6541078bfbe9b4289ae3abdd451`,
and exposure successor
`sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a`.
The sanitized record is
[`data/atomic_smoke_n1_operational_failure_v1.json`](data/atomic_smoke_n1_operational_failure_v1.json).

Before that consuming attempt, a setup launch correctly failed because the
cache store was mode `0755`, not the required `0700`. It persisted no exposure
and consumed nothing. Every future atomic store must be a canonical,
non-symlink directory at mode `0700`.

### 0.1 Run attempt two only as the exact journaled successor

Attempt two is not a reroll. It must authenticate this complete lineage:

- historical A3 ledger:
  `sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4`;
- first-attempt incident file:
  `sha256:2cf35e733c9a392999ec904660b2b0bf17814c253e3936476023f3e815fc14ad`;
- first-attempt config:
  `sha256:9dad0a5f468d1e8f3c65f7b83ac1ce7d2072e6541078bfbe9b4289ae3abdd451`;
- first-attempt outer reason:
  `2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`;
- active predecessor exposure ledger:
  `sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a`.

The active predecessor must be exactly one canonical exposure append after
A3. The first selected task and every other predecessor ID stay excluded. The
remaining eligible universe is exactly nine IDs, digest
`sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed`.
A different predecessor, incident, lineage, count, or set digest stops before
selection.

The command must persist its secret-free config, then stage and authenticate
the pinned native launcher before generating selection, episode, or label
secrets and before exact-task exposure. It must use a fresh empty call-journal
directory at mode `0700`. The journal writes a bound header; for every one of
the fixed 29 call slots it writes the exact intent before transport and the
validated result before the next intent. The terminal is durable before the
runner returns. A pre-existing header, an open intent, or any partial prefix is
terminal evidence for that attempt, not permission to resume or retry it.

No attempt-two live outcome exists at this pre-live stage. Do not infer one
from passing tests or from the synthetic 29-call harness.

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
the corrected HD constituent-disjoint policy, the DRILL reservoir immediately
before A3 had exact maximum 24 = 24 `bd` + 0 `hd`; 48 was impossible. The earlier 28-unit
upper bound applied within-new-batch HD disjointness but failed to seed the
constituent-token exclusion set from the complete A2 ledger. Every remaining
DRILL HD pair shares at least one constituent with that projected set. A3 used
the frozen selector, one no-reroll seed, and the resulting seed-ranked greedy
capacity without changing score-bin edges, confidence, decision boundary,
families, or polarity after inspection. Its terminal record is in Section 3.

Exact v3 replay against A3's successor exposure ledger now certifies strict
DRILL capacity **zero** (`0 bd + 0 hd`, zero eligible tasks and groups), digest
`sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1`.
DEV remains exactly `16 bd + 0 hd`, digest
`sha256:434c0756e89891c4a10e31fdf0c97e2e9373930a2ed48e1ecfa011c36f15c4c8`.

The source receipt hashes every potentially authoritative Bongard Python
module but explicitly excludes the exact non-authoritative
`bongard/semantic_checker.py` sidecar. This realizes the un-Lean invariant:
installing, editing, or deleting an optional checker cannot change the receipt
identity, while any authoritative Python-source change still invalidates the
run.

The source boundary is not the whole process boundary. A post-A3 audit found
that `codex_launcher_digest` committed only the installed JavaScript wrapper;
the wrapper dynamically selected a separate native client. A3 therefore binds
the wrapper plus `codex-cli 0.146.0`, not exact native bytes. Before another
live run, the executable closure had to be fixed. That is now implemented: the
production boundary opens the native binary without following links, copies
and hashes the same descriptor into a private staged executable, verifies
`codex-cli 0.146.0`, and rechecks the staged identity after use. The pinned
native digest is
`sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02`.

A3 changed `minimum_clusters_per_bin` from 12 to 8 before its seed, pixels, or
model output. For two Bonferroni-adjusted bins at 90% confidence, eight gives
Hoeffding radius 0.480161 and is the smallest count that can possibly place an
interval wholly above or below the fixed 0.5 boundary. The run nonetheless
produced only six upper-bin clusters. The preregistration was honored; the fit
failed.

The larger-corpus audit also separates two quantities that had been conflated:

| population | exact-unused after A3 |
|---|---:|
| train + validation | 10,047 / 10,200 |
| `ff` | 2,998 |
| `bd` | 3,434 |
| `hd` | 3,615 |

Before the first atomic attempt, exactly ten of those tasks were exact-unseen
training tasks from already exposed Basic-shape generator clusters. That
historical frozen-universe digest is
`sha256:3246017440379de1e49f695503536f75062626d2de36bdab9112e96281e269a8`.
The first attempt exposed and consumed one selected task. The active successor
therefore has exactly nine remaining IDs, digest
`sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed`.
They are useful for a no-reroll transport/synthesis smoke because selecting one
consumes no new generator claim. They are not independent evaluation units.
The production successor precommit authenticates the full 12,000-task manifest,
the exact A3-to-`b053` lineage, and the first incident, selects from the nine-ID
universe metadata-only, and persists the exposure before hashing selected
pixels.

The pre-A3 24-unit ceiling and post-A3 zero are caused by demanding
constituent-disjointness across the complete predecessor ledger and the new
batch, not by lack of images. A3's
seed changed the selected BD representatives and their order, not the maximum.
The run was therefore BD-only and cannot support HD or mixed-family
generalization.
A next-generation calibration frame should admit exact-unused training tasks
while excluding reserved DEV/SEALED
semantic keys, treat shared generators as dependence rather than silently as
independence, and blind-score both held-out panels of each task before opening
either label. Evaluation must remain strictly held out. The current HD
pair-level partition cannot support constituent-disjoint evaluation and must
eventually be rebuilt at the attribute level.

## 3. Preserve A3 as a terminal scientific failure

A3 completed the headless-Codex transport path. It exited 2 with the exact
reason `calibration score bins are underpopulated: 1`. The command receipt is
`sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681`;
the terminal failure is
`sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb`.

| A3 measurement | count |
|---|---:|
| proposer calls / transport successes | 22 / 22 |
| accepted soft claims | 15 |
| direct-only attrition | 6 |
| typed-parser rejections | 1 |
| scorer calls / transport successes | 15 / 15 |
| scores `0 / 0.5 / 1` | `8 / 1 / 6` |
| lower bin `[0, 0.75)` | 9 clusters; 1 affirmative |
| upper bin `[0.75, 1]` | 6 clusters; 5 affirmative |

The fixed minimum was eight clusters per bin. The upper bin was short by two,
so no calibration was fitted and Stage B did not run. With only 15 scoreable
claims, two eight-cluster bins were impossible regardless of their scores.
This is a recruitment/bin-power failure.

Do not tell a negation story about A3. Intended-bin orientation was 13/15 and
its exact complement was 2/15. At the naive `score >= 0.5` threshold,
orientation was 12/15 and the complement was 3/15. Negation did not win.

The one parser rejection was also accidental: the forbidden-code regex matched
the prefix `def` in the ordinary cue word `defines`. The parser now requires a
complete forbidden-keyword match. This post-A3 correction does not rewrite the
terminal result or make the exposed cohort reusable.

A3 exposed 22 tasks. It leaves 10,047 exact-unused train/validation IDs (FF
2,998; BD 3,434; HD 3,615). SEALED/test remained untouched.

## 4. Stage B remains unauthorized

Neither the failed A1 receipt, the invalidated A2 incident, nor the failed A3
fit can authorize Stage B. There is no current successful Stage-A receipt from
which to build it.
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

## 5. Do not open SEALED

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

## 6. Fill the known perception and synthesis gap

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

For open categories, use an explicit operational observer instead of
pretending to formalize prose into truth. The atomic implementation now:

1. freezes twelve candidate-independent, neutrally named vision descriptions;
2. lets a text-only proposer derive 1--12 single-phrase affirmative atoms from
   those descriptions and support labels;
3. records the complete atom-by-panel observation matrix;
4. deterministically selects a positive conjunction of at most four atoms by
   support coverage, description length, and lexicographic tie-break;
5. excludes `Not`, polarity flips, and complement rescue; and
6. freezes the formula before query sources exist in the runner.

An uncalibrated `operational_nonmatch` can reject a panel only inside an
explicit operational archive whose calibration, semantic-truth, and benchmark
flags are all false. Its general truth projection is `indeterminate`; it cannot
be laundered into `certified_absent`. Calibrated-semantic atomic selection is
hard-disabled until Python can independently cold-validate a typed calibration
artifact and its interval rule.

The remaining representation gap is richer structured vision. One prose
sentence is still lossy. Add typed object, part, angle, topology, and relation
observations, with uncertainty and exact receipt bindings, so synthesis can
compare facts as well as opaque phrases. Separately recruit calibration tasks
label-blind through a frozen order until bins are powered, or freeze a batch
sized for measured attrition and occupancy.

Lean is not needed for any of this. The predicate, selection rule, evidence,
calibration, replay, and artifact identity remain Python-defined. A checker may
inspect a frozen result only as an optional removable sidecar.

## 7. Diagnose reverse predicates directly

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

## 8. Keep Python authoritative

All new predicate execution, calibration, selection, replay, benchmark
decisions, and scientific artifact IDs must have complete Python semantics.
Lean or another proof checker may only consume an already-frozen artifact as a
detached optional sidecar. Deleting that sidecar must change no result,
decision, or ID. There is no planned Lean migration and no Lean dependency on
the benchmark path.

## Completion criterion

The current record is complete as failure accounting: A1 failed, A2 was
invalidated without a terminal artifact, A3 ended in a canonical
underpopulated-bin failure after successful proposer/scorer transport, and the
first atomic N=1 ended as an operational wrapper failure with no recoverable
score. Before any new calibration is declared, a powered, label-blind
recruitment rule must be frozen. The atomic matrix path is implemented but not
itself a calibration. The consumed N=1 task may not be rerolled. Attempt two is
a distinct, incident-bound, nine-ID successor and has not yet produced a live
outcome. Any future separately authorized smoke must still be reported as
exploratory regardless of score. Stage B remains unauthorized. This phase is
not complete merely because the code builds, one smoke scores 2/2, or a
favorable complement can be found.
