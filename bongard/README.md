# Bongard visual-semantic benchmark

This directory contains the official-ShapeBongard visual track. Its purpose is
to test whether a headless Codex proposer can infer one affirmative rule from
six positive and six negative panels, pass a fresh exact support replay, and
then classify one held-out panel from each side.

There is no good benchmark result yet. The earlier raster-prototype baselines
failed their support gates; A1 terminated as an infrastructure/schema failure;
and A2 was invalidated by a concurrent source edit. A3 finally exercised the
complete headless Stage-A proposer/scorer transport chain, but terminated as a
canonical scientific failure because its upper calibration bin was
underpopulated. It did not reach an end-to-end Bongard benchmark.

## Current status: A1 failed; A2 invalidated; A3 failed; atomic N=1 failed operationally

The successor is implemented and offline-verified, but its first live N=1
attempt produced no Bongard result. Its synthetic causal harness completed all
29 model-receipt slots, made both query predictions, and cold-replayed; that
verifies the protocol only. The live command was launched from commit
`62ea577f5d86d109577f4f5e49b8b4866eb76c92`, tagged
`bongard-atomic-pre-smoke-20260806`. It durably persisted the cache snapshot,
secret-free config, and exact-task exposure, then failed to persist a terminal.
No prediction or terminal artifact exists. The selected task is consumed and
will not be rerolled.

The exact outer CLI error was `failed run precommit is not canonical JSON`,
with reason digest
`2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`.
The runner was entered and returned a typed `AtomicSmokeRun`. Fallback terminal
construction then tried to JSON-clone its frozen `MappingProxy` precommit and
failed. Normal terminal construction contains the same deterministic defect,
but the surviving outer error does not establish which exception first entered
the fallback path. The underlying run status, phase, output, and successful
model-call count are not recoverable. The only valid call-count statement is
**unknown, bounded 0..29**. Because no prediction was persisted, labels could
not be materialized or revealed. There is no score, calibration, semantic,
benchmark, or official-test claim. This is an operational failure, not a poor
or successful Bongard result.

The persisted content addresses are cache
`sha256:1094dfd6794d4dfd141b9d0d1c89cf648d5c7d57ea0a545868bc38df928f28a4`,
config
`sha256:9dad0a5f468d1e8f3c65f7b83ac1ce7d2072e6541078bfbe9b4289ae3abdd451`,
and exposure successor
`sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a`.
The sanitized machine record is
[`data/atomic_smoke_n1_operational_failure_v1.json`](data/atomic_smoke_n1_operational_failure_v1.json).
Its file SHA-256 is
`2cf35e733c9a392999ec904660b2b0bf17814c253e3936476023f3e815fc14ad`.
An earlier pre-exposure setup launch found a cache store at mode `0755`; the
required mode is `0700`. That launch persisted no exposure and consumed
nothing.

### A1 terminal record

A1 was a descriptive, exploratory calibration attempt on the clean DRILL
cohort. Its command receipt is
`sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`.
It terminated with scoring-failure digest
`sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`.

| A1 funnel | count |
|---|---:|
| selected candidates / successful proposer calls | 48 / 48 |
| accepted soft claims | 37 |
| direct-only attrition | 10 |
| typed-parser rejections | 1 |
| scorer transport errors | 37 / 37 |
| successful scores | 0 |

The labels remained withheld. A1 produced no scores, fitted calibration,
semantic-accuracy estimate, or evidence about whether negation helps. Its
protocol digest was
`sha256:861397b2fb9597ab6ad72b8993d7b032c6dfdce985840b30684a1d15d28a3c54`.
Its consumed no-reroll seed was
`f9ee0fc4433df603049734153ae5eeac7e7227873fd2f3f36bc163449f107857`,
and its durable exposure successor remains
`sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78`.
Those identities are historical and cannot be reused.

### A2 invalidation incident

A2 removes the provider-incompatible `minItems`, `maxItems`, and `uniqueItems`
keywords from the scorer transport schema; exact cue coverage, ordering,
uniqueness, and witness ownership remain fail-closed Python decoder checks.
This changes the protocol identity, so A2 is not an A1 retry.

| A2 field | frozen value |
|---|---|
| design | `descriptive-exploratory-only/v1` |
| candidate tasks | 48, from `bd` and `hd` |
| model | `gpt-5.6-sol`, medium reasoning |
| protocol | `sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca` |
| predecessor exposure ledger | `sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78` |
| fresh no-reroll seed | `eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1` |
| durable successor ledger | `sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8` |
| state | **INVALIDATED BY LIVE SOURCE MUTATION** |

A concurrent agent edited `bongard/typed_visual_proposal.py` after A2 froze its
protocol and cohort. The resulting grammar digest no longer matched the frozen
protocol, so the process exited without writing a Stage-A terminal artifact.
The incident record has file digest
`sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.
Process output showed only that 48 proposer and 34 scorer launches occurred;
their outputs were lost and are not scientific results. Labels were not
revealed. A2 supports no calibration, accuracy, or semantic inference, its
selected cohort remains consumed, and the same cohort may not be rerun.

### A3 terminal scientific failure

A3 ran from frozen source with a fresh no-reroll selection. It exited 2 with
the exact scientific failure reason `calibration score bins are
underpopulated: 1`. Its command receipt is
`sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681`
and its terminal failure is
`sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb`.

| A3 funnel | count |
|---|---:|
| selected candidates / successful proposer calls | 22 / 22 |
| accepted soft claims | 15 |
| direct-only attrition | 6 |
| typed-parser rejections | 1 |
| successful scorer calls | 15 / 15 |
| scores `0 / 0.5 / 1` | `8 / 1 / 6` |

The lower fixed bin `[0, 0.75)` contained 9 clusters with 1 affirmative label;
the upper bin `[0.75, 1]` contained 6 clusters with 5 affirmative labels. The
preregistered minimum was 8 per bin, so no calibration was fitted and Stage B
remained unauthorized. Intended-bin orientation was 13/15 versus 2/15 for its
exact complement. At the naive `score >= 0.5` threshold, orientation was 12/15
versus 3/15. **Negation did not win.** A3 exposed 22 tasks and leaves 10,047
exact-unused train/validation IDs: 2,998 `ff`, 3,434 `bd`, and 3,615 `hd`.
SEALED/test pixels were untouched.

The sole parser rejection also exposed a concrete bug: the forbidden-code
regular expression matched the prefix `def` in the ordinary word `defines`.
The boundary now requires a complete keyword match, so `defines` is accepted
while the actual Python keyword `def` remains rejected. This was fixed after
A3 and does not alter its terminal record.

A later executable-closure audit found an independent receipt limitation.
A3 committed launcher digest
`134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477`,
which is the installed JavaScript wrapper, plus reported version
`codex-cli 0.146.0`. That wrapper dynamically spawned a native client whose
bytes were not committed. The native file currently installed has digest
`sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02`
and size 271,056,976 bytes, but that is a post-hoc observation and cannot prove
which native bytes A3 executed. There is no evidence of mutation; there is
also no exact native-client authentication claim. New live runs must resolve,
execute, hash, and recheck the native binary itself.

Stage B did not run and is unauthorized by A1, A2, and A3. A metadata-only
post-A2 audit found that task count was the wrong capacity measure. Before A3,
after collapsing semantic siblings and projecting every complete-A2 exposure into
owner-independent HD constituent tokens, DRILL has an exact reservoir maximum
of 24 units: 24 `bd` and zero `hd`. The earlier 28-unit upper bound enforced HD
disjointness only inside the proposed new batch and failed to seed that token
exclusion set from the predecessor ledger. The original 48-task design is
impossible on the remaining frozen frame.
DEV has 16 `bd` units and **zero** `hd` units disjoint from the complete live
ledger. The Stage-B default of 24 therefore fails before pixels or model calls;
even a 16-task BD-only pilot cannot meet its preregistered 24-cluster minimum.
Any later experiment remains descriptive and hard-codes
`dependence_design_authorized = false`.

The exact v3 selector replay against A3's successor ledger reduces strict
DRILL capacity from that pre-A3 maximum of 24 to **zero**: zero eligible tasks,
zero eligible generator groups, and `0 bd + 0 hd`. Its capacity-certificate
digest is
`sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1`.
DEV remains exactly `16 bd + 0 hd`, certificate
`sha256:434c0756e89891c4a10e31fdf0c97e2e9373930a2ed48e1ecfa011c36f15c4c8`.

The pre-A3 24 and post-A3 zero are policy capacities, not the size of the
archive. After A3, 10,047 of
the 10,200 official train/validation task IDs remain exact-unused: 2,998 `ff`,
3,434 `bd`, and 3,615 `hd`. They are not all
semantically independent or certified clean. The current Stage-A selector
conflates calibration with strong holdout evaluation by forbidding every
constituent exposed by the predecessor ledger or shared inside a batch. A later
calibration-frame redesign should
use exact-unused training tasks, keep DEV/SEALED semantic keys reserved, model
shared-generator dependence explicitly, and score both held-out task panels
before either label is opened. The evaluation split must remain stricter; in
particular, HD must be partitioned by constituent attribute rather than by
ordered pair.

A3 fixed eight distinct clusters per bin before its seed, pixels, or model
output. That was the smallest count whose 90% simultaneous Hoeffding radius
(0.480161) could possibly decide against the fixed 0.5 boundary. The run then
produced only six upper-bin clusters. This is recruitment and power failure,
not evidence that the scorer's polarity is reversed. With only 15 accepted
claims, two bins of eight were mathematically impossible even before their
observed rates were considered.

New Stage-A command receipts are source-bound v2 records. The runner snapshots
the Python source boundary and checks it around exposure, every proposer and
scorer call, cold replay, and terminal serialization. A post-precommit source
change now writes a durable operational-failure artifact and failed receipt
with labels withheld. Canonical serialization caches preserve exact bytes and
digests while reducing the synthetic Stage-A replay from 161.15 s to 11.50 s;
the compact central-campaign Stage-B replay fell from 218.88 s to 51.10 s.

Visual-semantic SEALED/test execution is disabled both in the CLI and in the
direct benchmark runner. It must stay disabled after Stage B: the current
design does not support an inferential authorization claim.

## What the pipeline actually is

It is not `panel -> prose -> Lean -> truth`. Lean cannot certify that prose
matches pixels. The new atomic smoke instead makes the empirical boundary and
the logical boundary separate and replayable:

```text
12 support PNGs
  -> 12 isolated, neutrally named vision descriptions
  -> labelled descriptions only, with no pixels, to one atom proposer
  -> 1..12 affirmative single-phrase observer predicates
  -> 12 isolated one-panel scoring calls covering every atom
  -> complete atom x support-panel matrix
  -> deterministic positive conjunction of at most four atoms
  -> frozen formula

only after that freeze:
  2 query PNGs
  -> 2 isolated vision descriptions
  -> 2 isolated selected-atom scoring calls
  -> durable joint prediction commitment
  -> label reveal and score
  -> model-free cold replay of all 29 call receipts
```

That is the implemented successor to A3. In A3, the descriptions were
audit-only, the proposer made one irreversible bundled guess, and no complete
formula was evaluated. The atomic path now uses the descriptions as the sole
input to atom proposal, preserves every atom-by-panel result, and lets Python
select the formula. It removes the old minimum-over-cues collapse and every
post-hoc polarity flip.

The successful path has exactly 29 model calls: 14 neutral image-description
calls, one text-only atom proposal, and 14 one-panel atom scoring calls. Every
description binds the exact panel digest, support/query phase, protocol,
validated receipt, run commitment, and call ordinal. Every atom observation
also binds the scorer producer/version/method, output, receipt, run, and call
ordinal. The formula is frozen before a query source is read.

This makes the computation rigorous, not the English ontology. A phrase such
as `bird-like object` denotes a fixed operational observer question. Its
answer is reproducible under the frozen model, prompt, pixels, description,
and receipt. It is not a proof that the depicted object really is a bird.
Scientific construct validity still requires an independent calibration
artifact. The atomic calibrated-semantic mode is therefore hard-disabled
until the core can cold-validate such an artifact and its interval rule.

“Candidate-independent” has a narrow, testable meaning: the witness extractor
receives only one panel's PNG bytes. It does not receive the task ID, side,
label, proposed phrase, or formula. Every packet binds the source bytes,
extractor code, preprocessing choices, component ownership, and uncertainty.

The same condition holds for the atomic descriptions: each vision call sees
one neutrally named panel and no task ID, label, side, other panel, or candidate
phrase. Candidate phrases are proposed only after all twelve support
descriptions have been frozen.

The three retained preprocessing scenarios are:

- `threshold032.raw`;
- `threshold064.close-cross-1`;
- `threshold096.raw`.

The complete direct conjunction is evaluated separately inside each scenario.
Only then are the three outcomes combined. This preserves correlations such
as which hole belongs to which component; it does not manufacture independent
feature intervals.

## The direct catalog, exactly

The current catalog contains these ten atom families:

| catalog key | measured positive claim |
|---|---|
| `component.count` | exact number of separated ink components |
| `hole.owner_count` | exact number of enclosed regions with one component owner |
| `topology.endpoint_count` | exact number of open-stroke endpoints |
| `topology.branchpoint_count` | exact number of skeleton branchpoints |
| `topology.cycle_count` | exact number of stroke cycles |
| `topology.crossing_count` | exact number of certified four-arm X junctions |
| `curvature.reversal_count` | exact number of persistent signed-curvature reversals |
| `curvature.run_count` | exact number of persistent signed-curvature runs |
| `curvature.s_like_count` | exact number of simple open S-like strokes |
| `curvature.u_like_count` | exact number of simple open U-like strokes |

Every selectable target count is an integer from 1 through 8. Zero is excluded
because “count equals zero” is a negated existence claim disguised as an
equality.

That is the whole direct catalog. There is no implemented direct atom for
point contact, exterior gaps, owner-labelled contact rays, oblique-angle
bands, part correspondence, or bird-likeness. A previous narrative described
a complete two-loop point-contact signature; the executable catalog does not
contain it. Adding those capabilities is future perception work, not a
documentation synonym for what exists.

The crossing atom also has a limited meaning: it detects a geometric X in a
thinned raster graph. It cannot distinguish an over/under crossing from a
four-way attachment.

## What a soft predicate means

Soft prose is allowed because some concepts, such as “bird-like object” or
“mostly oblique parts,” are not in the direct catalog. The phrase does not
become ground truth or executable geometry. In the atomic path it becomes one
opaque, affirmative observer predicate:

1. Twelve candidate-independent vision descriptions are frozen first.
2. A text-only proposer sees those descriptions and their support labels and
   emits 1--12 single-phrase atoms. It sees no pixels.
3. A conservative surface guard rejects obvious negation, alternatives, and
   bundled cues. This guard is deliberately not advertised as a proof of
   logical atomicity.
4. A blind one-panel observer evaluates every atom with the same prompt and
   returns `present`, `operational_nonmatch`, `indeterminate`, or `error`.
5. Python keeps the complete matrix and deterministically chooses the
   shortest positive conjunction that presents every positive support panel
   and operationally rejects every negative one.

`operational_nonmatch` means only that the frozen observer returned nonmatch.
It may act as false inside an archive explicitly scoped to that observer. When
projected into the general semantic evidence lattice it is
`indeterminate`, never `certified_absent`. The archive permanently records
`calibration_authorized = false`, `semantic_truth_claim = false`, and
`benchmark_claim_authorized = false`.

The exact phrase, description, model, prompt, PNG digest, output, receipt,
run, call ordinal, matrix, selection rule, formula, and replay are rigorous.
Whether the phrase corresponds to a human category is an empirical question.
That is the correct boundary: Python can verify the computation over frozen
observations; neither Python nor Lean can prove the original visual judgment
from prose alone.

A1 was intended to estimate scorer behavior conditional on the proposer
emitting a soft claim, but all 37 scorer calls failed before producing a score.
It therefore estimated nothing about semantic correctness or scorer accuracy.
A2 was a separate repaired-protocol experiment, but live source mutation
invalidated it before a terminal artifact was written. Its observed launches
and lost outputs provide no scorer or semantic estimate.

A3 repaired the transport path: all 15 scorer calls succeeded. It still did
not fit a calibration because the high-score bin had only 6 clusters instead
of 8. Its positive orientation was strong (13/15 by intended bin; 12/15 at the
naive threshold), so its failure cannot honestly be summarized as “negation
wins.”

## Four dispositions, without word games

- `present`: the frozen affirmative predicate has constructive evidence.
- `certified_absent`: the affirmative predicate has an operationally certified
  nonmatch under a frozen direct or independently calibrated semantics. The
  new uncalibrated atomic scorer cannot emit it.
- `indeterminate`: retained scenarios disagree or the evidence interval cannot
  decide the threshold.
- `error`: extraction, transport, parsing, identity, or replay failed.

A fifth serialized value exists only inside operational atomic archives:
`operational_nonmatch`. It is an exact observer response, not a fifth semantic
truth value. Its general semantic projection is `indeterminate`. A failed fit
or failed model call is never a negative. `indeterminate` and `error` count as
incorrect in headline accuracy; they are also reported separately.

## Why reverse alignment appeared in earlier diagnostics

The earlier PURE support-prototype diagnostic recorded 10 reversed outcomes
among 132 executable support-panel outcomes. That was an alignment diagnostic,
not an A1 negation experiment and not permission to execute a complement.

If the complement of a synthesized rule scores better, the synthesized rule
has learned the wrong orientation or a spurious correlate. That can happen
even when the English phrase sounds plausible:

- twelve support panels underdetermine the concept;
- a feature can separate one generator sample while reversing on another;
- pooled `bd`/`hd` selection can reward family or attribute reuse instead of
  concept transfer;
- a parser, fit, or observer failure can look like Boolean false if the
  evidence type collapses failures into negatives;
- post-hoc polarity search can rescue a bad predicate by flipping it after it
  has seen the support outcomes.

The old statistical story also treated reused generator attributes too much
like independent examples. Its claimed eight HD DEV units were only distinct
as exact ordered pairs; they were not disjoint from prior constituent
attributes. Replaying the complete A2 ledger leaves 16 `bd` and zero `hd` DEV
units. The historical mixed-family total of 24 was therefore not a valid
strict-disjoint capacity claim.

The current path removes the cheap rescue: the proposer must state the
positive-side rule, the IR has no `Not`, all selected atoms are used once in a
conjunction, and the exact support gate rejects a reversed rule before query
pixels are released. This makes failure honest; it does not make perception
good. Better results require better panel observables and a better calibrated
visual scorer.

The atomic smoke now implements the Python-native description/matrix/selection
path: one-phrase atoms, complete atom-by-panel observations, deterministic
small positive conjunctions, no `Not`, and no polarity flip. The remaining
perception problem is to replace lossy free prose with richer typed object,
part, angle, and relation observations without leaking task context. A future
semantic claim also needs a powered, independently frozen calibration design;
the exploratory operational path does not pretend to supply one.

## Python is canonical; Lean is removable

Python alone is authoritative for predicate construction and execution,
typechecking, calibration, selection, cold replay, benchmark decisions,
persistence, and scientific artifact IDs. Canonical JSON is the interchange
format.

Lean is not required and is not on the execution path. A future Lean or other
proof checker may independently inspect an already-frozen artifact, but it must
remain a detached optional sidecar:

- no Lean term may enter a predicate or artifact identity;
- no benchmark result may depend on Lean being installed;
- deleting the checker must leave results, benchmark decisions, IDs, admission,
  and replay unchanged;
- a Lean proof may establish consequences of recorded evidence, never the
  correctness of the original visual observation.

This is the explicit “un-Lean” portability invariant.
The Stage-A source identity enforces it mechanically: it hashes every
potentially authoritative Bongard Python module while excluding only the exact
non-authoritative `bongard/semantic_checker.py` sidecar boundary. Editing or
deleting that sidecar cannot change a command-receipt ID; changing an
authoritative module does.

## Corpus boundary

The pinned ShapeBongard V2 release has 12,000 tasks and 168,000 panels:

| family | tasks |
|---|---:|
| `ff` | 3,600 |
| `bd` | 4,000 |
| `hd` | 4,400 |

This is the complete public Bongard-LOGO release, not the earlier project
subset. The [upstream dataset repository](https://github.com/NVlabs/Bongard-LOGO)
also records 12,000 tasks in these three families and pairs every image with
an action program. Those programs and rule-bearing paths remain verifier-side;
the visual proposer never receives them. A recent
[symbolic-grounding diagnostic](https://arxiv.org/abs/2604.21346) reports a
large pixels-versus-structured-input gap on Bongard-LOGO, which supports the
working diagnosis here: representation is the bottleneck, not Lean.

The primary split is 9,300 train, 900 validation, and 1,800 test. The pinned
content identities are:

- archive: `sha256:8c5542ac7b9ce8a6a14d157a0656dbde9da5b7843424eade4bd653759d9a27d0`;
- split: `sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230`;
- extracted corpus: `sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138`;
- release descriptor: `sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b`.

The complete decoding audit found 168,000 single-frame RGB 512x512 PNGs and
zero anomalies. The official test pixels have not been used for semantic
model selection in this pipeline.

## Code map

- `visual_witnesses.py`, `contour_witnesses.py`, and
  `visual_witness_bundle.py`: exact-byte candidate-independent extraction.
- `visual_predicate_catalog.py` and `direct_visual_leg.py`: the finite direct
  catalog and registered Python atoms.
- `typed_visual_proposal.py` and `typed_visual_transport.py`: the closed
  support-only proposer boundary.
- `blind_soft_transport.py`, `soft_predicates.py`, and
  `family_soft_leg.py`: blind ordinal scoring and family calibration.
- `scenario_semantics.py`, `semantic_synthesis.py`, and
  `semantic_observation.py`: positive conjunction lowering and four-valued
  evaluation.
- `semantic_calibration_command.py`: write-once Stage-A command, durable
  exposure precommit, environment binding, and cold-verified receipt.
- `atomic_semantic_synthesis.py`: one-phrase atoms, exact atom-by-panel
  matrices, operational-versus-semantic scope separation, positive-only
  conjunction synthesis, and cold replay.
- `atomic_smoke_precommit.py`: complete-corpus authentication, metadata-only
  selection, and durable exposure before selected pixels are hashed.
- `atomic_smoke_runner.py`: the exact 29-call description/proposal/observer
  schedule, prediction persistence, label reveal, score, and model-free replay.
- `atomic_smoke_command.py`: source-frozen, native-launcher-authenticated,
  no-reroll production command boundary for one exploratory train smoke.
- `semantic_gated_dev_validation.py`: strict-disjoint descriptive Stage B.
- `semantic_run_verification.py`, `semantic_commitment.py`, and
  `artifacts.py`: exact archive reconstruction and model-free replay.
- `benchmark.py` and `cli.py`: freeze/query/reveal runner and external command
  boundary, including the visual-semantic test/SEALED hard stop.

## Local verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r bongard/requirements.txt
.venv/bin/python -m pytest -q bongard/tests
```

Do not run a live calibration merely to test the command. Live runs consume
the exposure ledger before model access. Use the synthetic tests for command
and replay validation.

The atomic command is likewise consuming: it durably records the selected
exact task before reading its selected pixels. Its production entry point is:

```bash
python -B -m bongard.atomic_smoke_command \
  --corpus /absolute/path/to/ShapeBongard_V2 \
  --archive /absolute/path/to/ShapeBongard_V2.zip \
  --exposure-ledger /absolute/path/to/a3-successor.exposure.json \
  --config-store /absolute/path/to/config-store \
  --exposure-store /absolute/path/to/exposure-store \
  --prediction-store /absolute/path/to/prediction-store \
  --terminal-store /absolute/path/to/terminal-store \
  --cache-store /absolute/path/to/cache-store
```

All stores must already exist as canonical, non-symlink directories with mode
`0700`. The CLI
prints one ID-redacted JSON status line. This run remains an exploratory
repeated-generator train smoke even if both query predictions are correct.

For an operational live command, invoke the production CLI from a detached
immutable commit with a newly created empty `PYTHONPYCACHEPREFIX` and
`python -B`. The source receipt hashes `.py` bytes; this launch discipline
prevents an older or crafted `.pyc` from becoming the executed program. The
dependency-injection parameters on the Python functions are test seams, not an
adversarial command authority, because they can substitute transports,
verifiers, or a watched source root. Stage B independently rechecks the
successful source identity.

See [CONTINUATION_PLAN.md](CONTINUATION_PLAN.md) for the next decisions and
[HISTORY.md](HISTORY.md) for falsified baselines and removed narratives.
