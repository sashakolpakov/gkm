# Bongard visual-semantic benchmark

This directory contains the official-ShapeBongard visual track. Its purpose is
to test whether a headless Codex proposer can infer one affirmative rule from
six positive and six negative panels, pass a fresh exact support replay, and
then classify one held-out panel from each side.

There is no good benchmark result yet. The earlier raster-prototype baselines
failed their support gates, and the first visual-semantic calibration
experiment, A1, terminated as an infrastructure/schema failure. A distinct
repaired-protocol experiment, A2, was then invalidated by a concurrent agent
source edit after its protocol and cohort were frozen.

## Current status: A1 failed; A2 was invalidated

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

Stage B did not run and is unauthorized by both A1 and A2. A metadata-only
post-A2 audit found that task count was the wrong capacity measure. After
collapsing semantic siblings and projecting every complete-A2 exposure into
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

That 24 is a policy capacity, not the size of the archive. Of the 10,200
official train/validation tasks, 10,069 exact task IDs remain absent from the
complete A2 ledger: 2,998 `ff`, 3,456 `bd`, and 3,615 `hd`. They are not all
semantically independent or certified clean. The current Stage-A selector
conflates calibration with strong holdout evaluation by forbidding every
constituent exposed by the predecessor ledger or shared inside a batch. A later
calibration-frame redesign should
use exact-unused training tasks, keep DEV/SEALED semantic keys reserved, model
shared-generator dependence explicitly, and score both held-out task panels
before either label is opened. The evaluation split must remain stricter; in
particular, HD must be partitioned by constituent attribute rather than by
ordered pair.

For the separately identified A3 engineering run, the seed will be generated
only after source freeze. Its candidate count will be the deterministic
seed-ranked capacity, with no reroll. Its two fixed calibration bins will
require eight distinct clusters each rather than the A1/A2 value of twelve.
At 90% simultaneous confidence this gives radius 0.480161, so eight is the
smallest bin that can possibly decide against the fixed 0.5 boundary. This is
a preregistered capacity repair made before A3's seed, pixels, or model output;
it weakens precision and does not authorize Stage B or SEALED.

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

It is not `panel -> prose -> Lean -> truth`. The implemented path is:

```text
6+6 labelled support pixels
  -> one typed affirmative proposal
       0..3 registered direct atoms
       0..1 soft claim, with 1..4 positive cues

each neutral panel's exact PNG bytes
  -> candidate-independent Python witness bundle
  -> direct atoms evaluated inside each correlated preprocessing scenario
  -> optional blind one-panel ordinal soft score
  -> valid, separately fitted development calibration interval
  -> one of present / certified_absent / indeterminate / error

all atoms
  -> closed positive conjunction in Python IR
  -> exact 12/12 support gate
  -> frozen formula and registry
  -> query release, joint prediction commitment, label reveal
  -> exact-byte cold replay without model calls
```

“Candidate-independent” has a narrow, testable meaning: the witness extractor
receives only one panel's PNG bytes. It does not receive the task ID, side,
label, proposed phrase, or formula. Every packet binds the source bytes,
extractor code, preprocessing choices, component ownership, and uncertainty.

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
“mostly oblique parts,” are not in the direct catalog. The prose does not
become ground truth. It becomes a frozen operational measurement protocol:

1. The support-only proposer emits one affirmative claim and one to four
   affirmative cues. It cannot emit code, thresholds, weights, `Not`, a
   complement, or a final Boolean.
2. For each panel, an isolated scorer sees neutral `query.png`, the frozen
   claim/cues, and short verifier-produced witness summaries. It does not see
   task identity, Bongard side, support/query role, source path, or label.
3. For every cue it emits exactly `supported`, `ambiguous`, or `unsupported`.
   Python maps these to 1.0, 0.5, and 0.0, and takes the fixed minimum over
   cues. A supported or ambiguous cue must cite a listed witness ID.
4. If a valid development calibration has been fitted, Python maps the score
   through it and evaluates the fixed affirmative probability boundary.

The witness citation proves only that the response refers to a witness the
verifier supplied. It does not prove that a panel is bird-like. The resulting
claim is rigorous only in the operational sense: exact text, cue inventory,
model/prompt, PNG digest, response receipt, ordinal map, aggregation,
calibration bin, interval, and evaluator are all fixed and replayable.

A1 was intended to estimate scorer behavior conditional on the proposer
emitting a soft claim, but all 37 scorer calls failed before producing a score.
It therefore estimated nothing about semantic correctness or scorer accuracy.
A2 was a separate repaired-protocol experiment, but live source mutation
invalidated it before a terminal artifact was written. Its observed launches
and lost outputs provide no scorer or semantic estimate.

## Four dispositions, without word games

- `present`: the frozen affirmative predicate has constructive evidence.
- `certified_absent`: the affirmative predicate has an operationally certified
  nonmatch under the frozen direct/calibrated semantics.
- `indeterminate`: retained scenarios disagree or the evidence interval cannot
  decide the threshold.
- `error`: extraction, transport, parsing, identity, or replay failed.

A failed fit or failed model call is never a negative. `indeterminate` and
`error` count as incorrect in headline accuracy; they are also reported
separately.

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
