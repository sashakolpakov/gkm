# Bongard: visual concept induction with verifier-gated growth

This directory contains the canonical Bongard experiment. The research target
is not merely to classify familiar panels. It is to grow a reusable library of
typed perceptual observations while preventing support-set tricks, polarity
flips, and parser failures from masquerading as concepts.

The current code establishes the corpus, information boundary, evidence type,
closed predicate language, immutable run artifacts, and promotion gates. A
full official-corpus score is not reported yet. Pre-rewrite symbolic and
rendered pilots are preserved at the annotated Git tag
`pre-bongard-complete-rewrite-20260805`; none is an official ShapeBongard
benchmark result.

## The pipeline, without euphemisms

```text
panel pixels
  -> frozen visual scorer or categorical judgment + exact receipt
  -> calibrated predictive interval or archived empirical outcome
  -> typed, provenance-bearing witnesses with uncertainty
  -> closed positive predicate (canonical HYBRID: one Atom; general IR: And / justified Or)
  -> conditional mechanical verification
  -> nuisance, calibration, near-miss, anti-memorization checks
  -> full accepted-archive replay
  -> atomic promotion of a new reusable leg
```

The important word is **conditional**. The verifier can prove that a frozen
predicate follows from the recorded witnesses and intervals. It cannot prove
that a vision model correctly saw a bird, an oblique angle, or a contact in the
original pixels. That perceptual claim remains empirical and must earn its
status through calibration, nuisance tests, near misses, fresh tasks, and
replay.

Prose has one legitimate role here: it names and specifies a candidate visual
claim. It is not itself the predicate and it is not evidence. The development
soft-predicate bridge consumes a distinct `FROZEN_VISUAL_SCORE` packet. Given
an externally admitted score, a caller-supplied expected plan digest, exact
development identities, and an externally justified cross-cluster sampling
assumption, it computes cluster-level predictive-support bounds. It checks
content identities and clustering constraints; it does not authenticate the
pixels-to-score operation, scorer or annotation receipts, publication time, or
independence between clusters. The current headless HYBRID path instead
archives uncalibrated categorical model judgments for a task-local frozen
claim. Its 12-panel support replay gate checks orientation and same-support
consistency; that gate is not calibration and does not establish
generalization.

The operational semantics today are pure Python: registered perceptual legs
are Python callables, `ir.py` typechecks and evaluates the closed formula, and
`artifacts.py` replays committed atom evidence from plain JSON without calling
a model or leg. The serialized IR, registry snapshot, and four-disposition
evidence are the backend-neutral contract. `predicate_backend.py` exposes that
boundary explicitly and selects the Python reference backend by default.

Lean is neither required nor currently used. If a Lean checker is added, it
must be an optional independent cross-check of the same serialized contract;
benchmark execution, admission, and cold replay must continue to work without
Lean. It still could not turn a fallible visual description into pixel-level
truth. Do not embed Lean terms in leg or formula identities, and do not make a
Lean proof a prerequisite for replaying an otherwise valid run.

## Start here

Create an environment and run the canonical unit suite:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r bongard/requirements.txt
.venv/bin/python -m pytest -q bongard/tests
```

The canonical package surface is:

- `corpus.py` — validated, content-addressed access to the complete
  `ShapeBongard_V2` archive or its equivalent generator layout;
- `release.py` — exact verification of the archive, split, task inventory, and
  extracted corpus against the checked-in official-release descriptor;
- `image_audit.py` — bounded-memory PNG verification, full-frame decoding,
  observed-property distributions, and explicit strict property checks;
- `cohorts.py` — metadata-only planning over historically clean official task
  cohorts;
- `exposure.py` — append-only exposure history and deterministic
  drill/development/sealed partitions;
- `evidence.py` — the only four runtime dispositions and provenance-bearing
  soft observations;
- `legs/contracts.py` — typed units, domains, codomains, transformations, and
  exact version-and-digest-pinned leg registration;
- `legs/bilateral_symmetry.py` — a candidate-independent deterministic
  reflected-ink measurement with a fixed preprocessing-sensitivity interval;
- `support_prototypes.py` — support-only interval feature centroids and a
  fixed positive contrastive margin, ready for later runner integration;
- `ir.py` — the closed positive predicate language and interval-safe
  evaluator;
- `predicate_backend.py` — the backend interface and pure-Python reference
  evaluation/replay implementation; a proof assistant can only be an optional
  conformance checker;
- `transport.py` — isolated headless-Codex image transport with byte-bound
  request and response receipts;
- `proposer.py` — the support-only visual proposer and four-disposition query
  observer;
- `synthesis.py` — compilation of a frozen positive visual claim into a
  content-addressed registered leg;
- `artifacts.py` — support commitment, proposal freeze, two-query prediction
  commitment, label reveal, tamper checks, and model-free cold replay;
- `benchmark.py` — verifier-controlled support/freeze/query/reveal episodes
  and scoring;
- `admission.py` — typed attachment, archive preservation, novelty accounting,
  and all-or-nothing promotion;
- `campaign_report.py` — fail-closed aggregation of bounded headless campaigns,
  including exposure-chain, receipt, identity, and stage-yield checks;
- `cli.py` — official inventory, cohort planning, one-episode execution, and
  externally anchored cold verification.

`benchmark.py` is the episode-level integration point: it gives the proposer
six positive and six negative support panels and fixes one proposal. Twelve
fresh, isolated, neutral single-image calls then replay that fixed claim on the
support panels. Only an exactly aligned gate is bound into the final proposal
freeze; the runner then releases two neutral query panels and commits both
predictions before the labels are revealed. A successful canonical episode
therefore makes one proposal call, twelve support-replay calls, and two query
calls. Query objects expose only neutral callback identifiers and a temporary
`query.png` path, never source filenames, corpus paths, task IDs, or labels.

## Complete official corpus and exact release identity

The target archive has 12,000 tasks and fourteen PNGs per task: seven positive
and seven negative. A complete extracted corpus therefore contains exactly
168,000 panel PNGs.

| family | tasks | role |
|---|---:|---|
| `ff` | 3,600 | free-form |
| `bd` | 4,000 | basic shape concepts |
| `hd` | 4,400 | abstract/compositional concepts |

The official primary split is 9,300 train, 900 validation, and 1,800 test.
The test split is also normalized into `FF` (600), `BA` (480), `CM` (400),
and `NV` (320). “Novel” in an upstream split name does not override our own
exposure history: if this repository, a person, or a proposer has already seen
a task, it is not unseen for our experiment.

The small public generator/gallery checkout previously used by this repository
is not the complete archive. Do not report a full benchmark from it.

The checked-in descriptor
[`data/shape_bongard_v2_release_v1.json`](data/shape_bongard_v2_release_v1.json)
pins these exact identities:

| object | bytes | SHA-256 content address |
|---|---:|---|
| `ShapeBongard_V2.zip` | 1,762,748,636 | `sha256:8c5542ac7b9ce8a6a14d157a0656dbde9da5b7843424eade4bd653759d9a27d0` |
| `ShapeBongard_V2_split.json` | 442,720 | `sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230` |
| sorted 12,000-task ID inventory | — | `sha256:4503ae6b40dc7b34520eb5b8a4cca6ff8153635df0f42db5f6715cc349602dd0` |
| extracted corpus manifest | — | `sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138` |

The descriptor itself has canonical digest
`sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b`
and pins upstream commit `9df7c78ee9c6a2ff041b48d9ed407359aac259c3`.

Structural completeness is not release identity. Verify the archive, exact
split bytes, task IDs, and content-addressed extracted tree without copying
either large input into Git:

```bash
.venv/bin/python -m bongard inventory \
  --corpus downloads/ShapeBongard_V2 \
  --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
  --require-complete \
  --official-release \
  --archive downloads/ShapeBongard_V2.zip \
  --out results/bongard/official-inventory.json
```

This command fails unless the archive filename, byte count, and SHA-256 match;
the split filename, byte count, and SHA-256 match; the family/split/regime
counts match; the sorted task-ID digest matches; and hashing every extracted
panel reproduces the pinned corpus-manifest digest. Its canonical JSON result
reports `task_count`, family and split counts, `manifest_digest`,
`split_source_digest`, and an `official_release` object containing the
descriptor, archive, task-inventory, and corpus-manifest commitments.

The loader itself remains useful for structurally valid non-official corpora:

```python
from bongard import ShapeBongardCorpus

corpus = ShapeBongardCorpus.discover(
    "downloads/ShapeBongard_V2",
    split_file="downloads/ShapeBongard_V2/ShapeBongard_V2_split.json",
    require_complete=True,
)
manifest = corpus.build_manifest()
print(manifest.digest)
print(len(corpus.tasks_in_split("test")))
```

The loader accepts the released `images/` layout and the equivalent `png/`
generator layout. It verifies task structure, seven panels on each side, PNG
signatures, official counts, split disjointness, and content digests. Only the
descriptor-backed command above establishes that those digests are the pinned
official release. Paths are never part of a scientific identity; bytes are.

## Full PNG audit: observe first, require second

Release verification binds all 168,000 compressed panel byte strings. The
separate image audit checks the decoding boundary: each source must be a
non-symlink regular file containing one exact PNG container with no trailing
data; Pillow must verify it and load every frame; and mode, width, height,
metadata/info keys, and frame count are recorded. Source bytes are read and
hashed once into a bounded-memory spool, so verification and decoding use a
frozen snapshot. A final filesystem pass detects ordinary replacement or
mutation during the audit.

The complete exploratory pass and an independent strict pass both completed on
the pinned release. The strict report is checked in as
[`data/shape_bongard_v2_image_audit_v1.json`](data/shape_bongard_v2_image_audit_v1.json).
The result is exact:

| property | complete-corpus result |
|---|---:|
| tasks / panels | 12,000 / 168,000 |
| total compressed panel bytes | 1,948,958,314 |
| container / mode / size | PNG / RGB / 512×512 |
| frames / Pillow info keys | 1 / none, for every panel |
| anomalies | 0 |

Its canonical report digest is
`sha256:d3485ada3605d708db82fbcfe6ecfc73506ce51ed85fcd1ce6ccd798e3bff9f8`.
The decoded-property summary digest is
`sha256:6feea60173c92a1357ffafbeecd78171c3455b3950a31229517ea07c6f03e811`;
the separately accumulated source-content summary digest is
`sha256:31f03303673e31a1a05f84ddd50621963ce4c73c1ab11073118c4905893389c5`.

The exploratory pass deliberately supplied no property guesses:

```python
import json

from bongard import ShapeBongardCorpus, audit_corpus_images

corpus = ShapeBongardCorpus.discover(
    "downloads/ShapeBongard_V2",
    split_file="downloads/ShapeBongard_V2/ShapeBongard_V2_split.json",
    require_complete=True,
)
manifest = corpus.build_manifest()
observed = audit_corpus_images(corpus, corpus_manifest=manifest)
print(json.dumps(observed.to_dict(), sort_keys=True, indent=2))
```

The data-only report contains task/panel/byte counts; family, format, mode,
size, info-key-set, and frame-count distributions; content and property
summary digests; the corpus-manifest commitment; a bounded anomaly sample;
and its own canonical digest. It contains no paths, pixels, or image objects.

The independent strict pass then supplied the observed values explicitly:

```python
from bongard import ImageExpectations, audit_corpus_images

confirmed = ImageExpectations(
    mode="RGB",
    width=512,
    height=512,
    info_keys=(),
    frame_count=1,
)
strict = audit_corpus_images(
    corpus,
    corpus_manifest=manifest,
    expected_properties=confirmed,
    require_expected_properties=True,
)
```

These are experiment inputs, not hidden library defaults. A non-strict pass
records mismatches as anomalies; strict mode raises `ImageExpectationError`
and exposes the completed report as `exception.report`.

## Exposure before sampling

Inspect the frozen historical-exposure classification before choosing tasks:

```bash
.venv/bin/python -m bongard cohorts \
  --corpus downloads/ShapeBongard_V2 \
  --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
  --require-complete \
  --split train \
  --cohort clean \
  --limit 50 \
  --out results/bongard/train-clean-cohorts.json
```

The cohort result gives its qualification, source-seed and split-index
digests, scope, counts, membership digests, and a bounded task-ID sample. It
is metadata-only: “historically clean” means not recorded by this repository's
frozen semantic exposure audit. It does not prove that official panel bytes
were unseen by people or absent from a foundation model's pretraining data.

On the complete release, that audit classifies 3,868 tasks as semantically
historically clean: 1,328 Basic tasks from unused shape families and 2,540
Abstract tasks from unused admissible attribute pairs. The 3,600 Freeform tasks
remain `indeterminate`, not “clean.” Partitions are made at the generator
concept level, not by task ID: all 20 instances of one Abstract ordered pair
stay together. The combined clean pool contains 2,769 drill, 542 development,
and 557 sealed tasks. The checked-in summary
[`data/shape_bongard_v2_cohort_summary_v1.json`](data/shape_bongard_v2_cohort_summary_v1.json)
has digest
`sha256:55de04a582ffa3a4fbf26466ab88f265ddd7839ae10004210cca4d9ffa4f8e9d`.

The frozen 2,769-task drill cohort is not a live availability count. Against
the checked-in sixteen-event campaign ledger head
`sha256:da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a`,
the resolver-v2 live overlay leaves 1,744 drill tasks and excludes 1,025
through semantic-key collisions; sixteen excluded tasks are also exact-task
collisions. The ledger records 22 task IDs and 38 semantic keys, producing 199
effective exposed keys after policy blocking. The overlay
digest is
`sha256:9e7ad95bc0fe2200d647c7ef9c34b81f8b041115265175be6fe63d6c67562dde`
and its live-membership digest is
`sha256:be680542b28a855d54cedcda6726d140af1ce4a8ad97c008511d5843f4e4b7e1`.
The ledger file is
`downloads/ShapeBongard_V2_full/exposure/abstract_006/da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a.exposure.json`.

The earlier resolver-v1 count of 2,609 was an overclaim: it treated numbered
Basic generator names as distinct families and therefore failed to exclude
obvious morphology siblings. Resolver v2 emits both the exact Basic generator
key and a conservative morphology key that removes a terminal number or
`_newN`; `advanced_lamp3`/`advanced_lamp4` and `bird2`/`bird7` consequently
collide. A cluster is blocked if a sibling was historically exposed or if
siblings cross the drill/development/sealed boundary. Its policy digest is
`sha256:48598ae580a2f88aee7652d36fd386d54a8e4265b040bf1313f558508f47af9a`.
For HD, “unseen” qualifies only the ordered attribute *combination*: the
component attributes may each be familiar, and the 20 generated instances of
one pair are siblings rather than independent unseen concepts.

Before the v7–v9 support releases, the initial resolver-v2 training
intersection contained 1,290 tasks across 161 retained semantic groups. At the
current head, the official train-and-drill scope contains 2,096 historically
clean tasks, of which 1,238 remain live and 858 are excluded, including sixteen
exact-task collisions; that overlay has
digest
`sha256:64c7f3cbd4444829d1bd8c50d1a99cc95d5830ec6459879a5a7f6668868eee90`
and live-membership digest
`sha256:2619ea03a9f32bddef941818791fee9d477040f073043da2c715547474813a23`.
The 506 other live drill tasks must not be consumed as training data. Resolver
collision domains are not statistically independent samples. The retrospective
montage disclosure and the v6–v12 support releases all remove their related
semantic groups. This overlay does not alter the frozen cohort or certify
unseen panel bytes. The run-time `external_anchor` fields are null; the later
Git commit is an after-the-fact publication anchor, not a preregistered ledger
anchor or proof that this was the latest authentic head.

Freshness is global state, not a command-line option. Import known historical
disclosures first, then partition only the remaining eligible task IDs:

```python
from bongard import (
    ExposureLedger,
    import_historical_exposures,
    load_historical_exposure,
)

historical = load_historical_exposure()
ledger = ExposureLedger.create(manifest.digest)
ledger = import_historical_exposures(
    ledger,
    historical.exact_official_task_ids,
    source="bongard/data/historical_exposure_v1.json",
    known_task_ids=corpus.task_ids,
)
ledger.write_content_addressed("results/bongard/exposure")
```

For an official CLI run, the exact-task ledger and semantic cohort check are
both required; one does not substitute for the other. With
`--require-unseen --cohort`, the runner checks exact and semantic collisions
and writes one task-level event before support release. It refuses the sealed
cohort outside `--sealed-test`. These checks do not intercept direct API reads,
manual inspection, or reuse of a stale ledger head. A prior task-level
disclosure makes every panel of that task non-unseen.

## Evidence is not Boolean

Every registered leg returns exactly one `Evidence[T]` disposition:

| disposition | meaning | may classify as negative? |
|---|---|---|
| `present` | the claimed value or witness was produced | no; it is evaluated by the atom |
| `certified_absent` | a declared procedure established non-existence | yes |
| `indeterminate` | the observation could not be resolved | no |
| `error` | implementation or contract failure | no |

A VLM may write “bird-like object” or “mostly oblique angles.” That becomes a
`SoftSemanticObservation`: the phrase, support interval, model/method, input
digests, and witness IDs are recorded. The object deliberately cannot be
coerced to `bool`. A calibrated atom may later assert, for example, that the
entire support interval is at least a fixed threshold. If the interval
straddles the threshold, the result is `indeterminate`, not whichever label is
convenient.

The generic HYBRID observer's wire outcome is `nonmatch`, not
`certified_absent`. Internally it can certify only the operational event “the
archived frozen-model procedure returned nonmatch for this claim.” It does not
certify that the depicted semantic property is absent from the pixels. A claim
of pixel-level absence requires a dedicated registered certifier or a
calibrated interval that lies wholly below its frozen threshold.

There is no universal image normalization. A leg declares which typed view it
uses: literal ink when stroke and rendering style are relevant, carrier shape
when those nuisances may be quotiented out, or a relational view for objects,
contacts, containment, repetition, symmetry, and ownership. Erasing stroke
width or raster detail globally would silently destroy concepts that depend on
them.

Two new development components make this boundary more concrete. The
bilateral-symmetry leg measures reflected-ink agreement directly from panel
bytes with fixed thresholds and a fixed reflection-axis search. It is
candidate-independent, preserves all four evidence dispositions, and registers
only the affirmative `AT_LEAST` direction. Its interval records preprocessing
sensitivity, not population calibration. Applied post hoc to v9, the score
does not recover that Bongard rule: multiple negative panels are more
bilaterally symmetric than positive panels. The missing representation is
therefore not simply “more symmetry.” It needs part/lobe ownership, the central
junction, and correspondence between the owned parts.

`support_prototypes.py` supplies the next task-relative layer without hiding a
polarity search. A neutral extractor first freezes panel-only interval feature
vectors without task ID, side, prose claim, or query role. The support fitter
then commits separate positive and negative centroids. Its only score is
`distance(query, negative) - distance(query, positive)`, so larger always means
more like the positive support; a fixed positive margin decides the predicate.
There is no side-swap or polarity-flip operation. This core is replayable but
is not yet wired into the official episode runner, does not extract pixels by
itself, and has not been externally calibrated.

## Why negation used to win

The old search had a structural escape hatch. It could fit a weak or inverted
feature on the labeled support panels, try the opposite polarity, and keep the
better orientation. Worse, failed fits and unrecognized structures sometimes
fell through to a false-like value. Negating that value converted lack of
perceptual competence into apparent negative evidence. With a tiny support
set, selection then rewarded the complement even though no positive concept
had been discovered.

This was not a mysterious model preference for negation. It was leakage from
the hypothesis language and the failure semantics.

The canonical primary track removes the escape hatch:

- there is no `Not` node and no polarity field;
- every scalar leg carries an `AffirmativeRelation` contract declaring which
  of `at_least`, `at_most`, or `between` means *more of its positive claim*;
- the IR rejects an inequality direction not declared by that leg, so a
  synthesizer cannot rescue a feature by trying its complement;
- a formula can contain only registered, digest-pinned static calls;
- the general proposal schema can represent positive atoms and conjunctions,
  but the canonical headless HYBRID CLI currently compiles exactly one
  `hybrid_claim` atom;
- explicitly justified disjunction exists in the verifier IR for retained
  library formulas, but the canonical proposer cannot synthesize it;
- an extractor failure is `error`, and uncertainty is `indeterminate`;
- only a constructive certificate produces `certified_absent`;
- units and closed intervals are checked mechanically;
- the proposal is frozen before query bytes are released;
- both query predictions are committed before either query label is revealed;
- near-miss, calibration, nuisance, anti-memorization, and archive replay gates
  are mandatory for promotion.

This does not guarantee a good predicate. It prevents a bad predicate from
being rescued by changing what “positive” means after seeing the labels.

`AffirmativeRelation` is part of the leg's signed, digest-bound interface, not
a threshold-search hint. A low-valued measurement can legitimately declare
`at_most`—for example, low closure error. In the reusable admitted-registry
track, that semantic direction and the registry must be frozen before support.
`LegRegistry` makes the direction immutable after freezing but does not enforce
that timing itself. The canonical HYBRID path is deliberately different: it
creates one task-local empirical leg after support, then freezes it before
support replay and query release. Non-scalar witness legs admit only `present`.

## Growing legs and gluing them

A **leg** is a reusable, typed observation procedure. Its contract states its
domain, codomain, unit, implementation digest, cost, and behavior under named
transformations. A formula references an exact registered version; arbitrary
Python and dynamic lookup are outside the primary language.

Growth follows a PowerPlay-like discipline:

1. Find a drill failure that the accepted registry cannot express or resolve.
2. Propose the smallest typed leg or composition that attaches at the existing
   boundary.
3. Test the attachment on constructive examples, nuisance transformations,
   calibration data, and deliberately close counterexamples.
4. Bind the stripped submitted candidate source to the registered leg's
   normalized source digest (or its selected bytecode fallback), require a
   direct formula call, and bind the comparison source to a verifier precommit;
   decoy novelty text fails. This is not a hash of dependencies, globals,
   closure contents, the package environment, or machine code.
5. Replay the candidate on its precommitted suite, then charge source novelty
   from normalized AST structure. Replacing code with a same-size rewrite is
   not free.
6. Replay every previously accepted attachment, not merely the new example.
7. Atomically promote only if every gate passes; otherwise the accepted
   archive remains byte-for-byte unchanged.

“Cofibrant gluing” is the architectural analogy: a new capability must declare
the typed interface along which it joins the retained library, and promotion
constructs the least compatible extension that preserves the archive. It is
not a license to accept decorative diagrams or proposer-supplied complexity.

Repeated compositions should eventually be factored into shared legs so
marginal novelty falls as the library improves. A benchmark score without the
growth trace, gate receipts, and archive replays does not establish that kind
of cumulative abstraction.

## Freeze, query, reveal, replay

The immutable artifact chain is:

```text
content-addressed support
  -> fixed positive claim + task-local formula/registry
  -> 12 isolated support observations + replay-gate artifact
  -> final formula/attachment/registry freeze binding the gate digest
  -> exactly two distinct unlabeled query blobs
  -> committed atom evidence + both predictions
  -> query-label reveal
  -> cold replay receipt
```

Each artifact contains or is bound by its parent's canonical SHA-256 digest.
Support and query bytes must be disjoint. Artifact-only logical replay
reconstructs the closed formula and cached atom dispositions from plain JSON;
it receives no registry implementation, vision model, or proposer callback.
Full CLI verification additionally reparses the proposal and canonically
recompiles the verifier-owned HYBRID formula, registry snapshot, source and
operational identities, and attachment contract without invoking the model.
Changing support, gate evidence, formula, query bytes, atom evidence,
predictions, or labels breaks verification.

The chain proves internal content integrity only when its root is anchored
outside the run file. Cold verification therefore requires the externally
recorded raw-file SHA-256; do not recompute that value from the same file at
verification time:

```bash
.venv/bin/python -m bongard verify \
  --run results/bongard/episode.json \
  --corpus downloads/ShapeBongard_V2 \
  --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
  --archive downloads/ShapeBongard_V2.zip \
  --expected-sha256 "$EXPECTED_RUN_SHA256"
```

`EXPECTED_RUN_SHA256` must be exactly 64 lowercase hexadecimal characters
copied from an independent write-once record, signed manifest, or committed
publication. Canonical verification additionally checks the trusted release
descriptor, exact split assignment and task manifest, a bijective mapping to
all fourteen official panel byte preimages, the canonical HYBRID compilation,
the support gate, every vision receipt's internal hashes and request/response
bindings, and cold logical replay. Those Codex receipts are not provider
signatures and may record that the JSONL omitted the reported model. The cold
verifier also cannot authenticate the unavailable history behind a non-empty
exposure-ledger predecessor. Proposal-failure records have no replayable
archive and fail.

## What counts as a result

Report three different things separately:

1. **Perceptual quality:** calibration, abstention/error rates, nuisance
   stability, and near-miss discrimination for each leg.
2. **Episode quality:** determinate and overall query accuracy on a declared,
   exposure-audited split.
3. **Growth quality:** new AST charge, reused legs, archive regressions, and
   accepted versus rejected attachments over time.

The protocol treats the sealed benchmark as one-shot. The CLI enforces unseen
status relative to the supplied ledger head, but cannot prove that the head is
latest or authenticate an unavailable non-empty predecessor; external
ledger-head anchoring and operator discipline remain necessary. Indeterminate
and error predictions score as wrong in headline accuracy, while their
dispositions remain visible for diagnosis. Threshold tuning, prose editing,
leg invention, and model selection belong to drill/development only.

Each current episode withholds exactly one positive and one negative panel.
If the caller-supplied seed makes query order uniform and independent of the
predictor, a no-vision strategy that merely assigns opposite labels to the two
opaque slots has 50% expected puzzle accuracy and 50% image accuracy. Puzzle
accuracy then does **not** have a 25% independent-label chance baseline. The
code deterministically derives order from the supplied seed; it does not
enforce randomness, secrecy, or external preregistration, and cold verification
has only the seed digest rather than its preimage. Reports must state this
conditional paired baseline and include per-image and per-side accuracy over
many independently selected semantic groups; a two-panel episode is only an
integration check.

A pre-current-protocol v1 drill on the training task
`bd_trapez_parallelogram_0000` classified its two held-out query panels
correctly (2/2). It used the old corpus identity and predates exact official
release binding, the current receipt/schema checks, canonical HYBRID
recompilation, and the 12-panel support gate. The checked-in complete-release
v2 completed drill scored 1/2.

The first checked-in current-protocol v3 drill attempt,
[`hd_exist_quadrangle-exist_sector_0000_v4.json`](runs/official_complete_drill_20260805/hd_exist_quadrangle-exist_sector_0000_v4.json),
has file SHA-256
`bf60a36bc7a48e61c61c8de2153753fa2996db54eacf53fe4c861bf06a9b4f41`.
It ended `support_rejected` before query release, with no query observations
or run archive and zero determinate query outcomes. All twelve support observer
calls otherwise succeeded, but the verifier converted every raw judgment to
`TransportIdentityError` because the supposedly stable
`cloud_config_bundle_cache_binding` changed across calls. Before that
conversion, the raw judgments fit 10/12 support labels: all six positives were
`present`, while four negatives were `nonmatch` and two were `present`.
This exposes a transport-identity protocol bug; it is not a held-out score.

The checked-in v5 drill attempt,
[`bd_advanced_lamp3_0000_v5.json`](runs/official_complete_drill_20260805/bd_advanced_lamp3_0000_v5.json),
has file SHA-256
`e3fbe8f76290bb93f33def26c36b50f9ae451e43456a52e4796976a71662255a`.
Its proposal and support calls share the single cache binding
`sha256:6860e08631caee1357061bd727e93f7d200931b3bb2d925f873aea3d669d22f2`,
so it clears the v4 transport-identity defect. It nevertheless ended
`support_rejected` before query release, with no query observations or run
archive and zero determinate query outcomes. The immutable artifact's formal
v3/v1 gate records six forward matches, one reverse match, and five parser
errors.

The raw model outputs were six positive `present` judgments, five negative
`nonmatch` judgments, and one negative `present` judgment. The five
`nonmatch` outputs exposed a prompt/schema/parser contradiction: their
top-level `reason` was allowed or inferred, while the archived parser required
it to be null. Current observation schema v4 and support policy v2 make that
field optional and bind it into the nonmatch certificate. Re-evaluating the
archived raw outputs under that repaired contract is only a post-hoc
diagnostic: it gives 11 forward matches and one reverse match, hence
`unsupported`. It does not rewrite or salvage the v5 artifact, whose archived
result remains `observer_failure`.

Even under the intended parse, the proposed “bent double-ended arrow” rule is
overbroad. An explicitly oracle-only post-hoc inspection of privileged Basic
action programs shows that the positive class uses one precise nine-action
template, while the false-positive negative is a distinct near-miss geometric
program. Those programs were not available to the proposer or gate and cannot
be used as benchmark evidence. The diagnosis instead sharpens the architecture:
prose proposes a candidate, while a frozen visual contour/template or prototype
scorer must operationalize it from pixels.

The upstream sampler definitions explain a broader failure mode, again only as
a post-hoc oracle diagnostic. Basic multi-shape tasks and Abstract
attribute-pair tasks construct the positive side as a conjunction. Their
negative panels can be several near-miss subgroups, each violating a different
positive conjunct. The proposer received neither task IDs nor action programs,
so it had to infer that structure from pixels. Its current prompt now demands
that every proposed cue cover every positive panel and that each negative fail
at least one cue; it explicitly warns against collapsing distinct near-miss
subgroups into one vague resemblance word. This prompt change is a hypothesis
about proposal quality, not evidence that the visual predicates are solved.

The subsequent v6 attempt selected the first lexicographic task in the
then-current live-eligible drill list,
[`bd_advanced_lamp4-exist_quadrangle_five_lines12_0000_v6.json`](runs/official_complete_drill_20260805/bd_advanced_lamp4-exist_quadrangle_five_lines12_0000_v6.json),
before any of its pixels had been inspected. That selection is
inspection-unbiased but deterministic, not random. The file SHA-256 is
`6a120eabd4efeeee60b5555cbb581d6cced3d33206bb0ed556e61a29fb213057`.
Its support-release event
`sha256:b8fe3ea944d118058ac52e6f849ab5c1c1f6e08737f155e8b23f87569610877a`
advanced the ledger to an intermediate ten-event head.

V6 ended `proposal_error` through exactly
`plan_committed -> support_released -> proposal_failed`. It has no accepted
proposal, support gate, query observation, or run archive. The archived failure
reports a blanket lexical rejection in category
`negative morphological complement`. This is a benchmark-attempt failure,
not a query score, and must remain visible in attempt coverage.

The old run schema did not preserve the rejected raw proposer payload or its
receipt, so the lexical decision cannot be independently audited or replayed.
That is a separate auditability gap. A future protocol must distinguish
constructive morphological descriptions from logical negation and persist
rejected-attempt payloads and receipts. Such a corrected run would be a new
attempt; v6 itself cannot be salvaged.

The v7 artifact,
[`bd_arc_cup_0000_v7.json`](runs/official_complete_drill_20260805/bd_arc_cup_0000_v7.json),
has file SHA-256
`9801dbec0928f59667993a993b99f2cfcd6d5c02264bb10ef467ac98c427a462`.
Its support-release event
`sha256:dbd578e1d3951837f25378721cf61e664eb96240e8f7c3fc108d1ff1db280a21`
produced ledger successor
`sha256:fc82fcebf4686c36f85f9efa0944ef4fc57b5da41dfccb19126c33b372c146dc`.
V7 ended `proposal_error` because DNS resolution failed before Codex returned a
response. It has no rejected proposal attempt, accepted proposal, support gate,
query observation, or run archive. This is a recorded transport failure, not a
negative prediction or score.

The v8 artifact,
[`bd_asymm_bridge_0000_v8.json`](runs/official_complete_drill_20260805/bd_asymm_bridge_0000_v8.json),
has file SHA-256
`ef50e35732c9a02d933ca1d7628589071270b06bc3d87fd0bb2543cdff16ccdb`
and status `complete`. The proposer named the claim “An enclosed region has a
glyph-decorated boundary.” Its frozen operational rule was: “The panel contains
an enclosed cell, and part of that cell’s boundary is rendered as a sequence of
repeated small geometric glyphs such as circles, squares, triangles, or zigzag
teeth.” All twelve isolated support replays aligned (6/6 `present` positives and
6/6 model `nonmatch` negatives), and both query predictions matched their
revealed labels. The artifact records the complete phase chain from
`plan_committed` through `cold_replay_verified`; cold verification checks all
fourteen panel-byte preimages. Its run-archive digest is
`4f679fe175383a3ceb85333bf85f644dbe2a1ab69033747ae4b7d133893dc2ef`
and its artifact-chain digest is
`c2cefb76126cc18d5f5b4e39c4b506fc259cb6fdb02ebf1a7dfa666f92631f4d`.
The support-release event
`sha256:25317bb78b0cf60b7585f59c93c7331c0f6743c3553ae044008b14b69d76fd35`
produced intermediate twelve-event successor
`sha256:7cf70dcb4e15aa8f0d8f82f4e5ff1e32f3018fb1f467061a5c947b0a5cf742d3`.

V8 is one successful integration episode, not an accuracy estimate. Its two
queries are one positive/one negative pair with the conditional 50% no-vision
baseline described above. More importantly, canonical HYBRID is still an
uncalibrated categorical self-observer: the same model family proposes prose
and judges whether each panel matches it. The archive proves ordering,
identity, and replay of those judgments; it does not prove their pixel-level
truth. The missing pixels-to-score leg—frozen, externally calibrated visual
measurements that make “bird-like,” “oblique,” contour-template, and similar
claims operational—is still the central theoretical hole.

The schema-v4 v9 artifact,
[`hd_balanced_two-symmetric_transposed_0000_v9.json`](runs/official_complete_drill_20260805/hd_balanced_two-symmetric_transposed_0000_v9.json),
is an official-training HD ordered-combination attempt with file SHA-256
`6171b6bca42ffa6423d0e7e1ef753da325ef3d000e6f39d2ca28b5afccf8e655`.
It proposed “A matched opposing pair of lobes joined at one center,” with
operational cues `paired_lobes`, `matched_geometry`, `central_junction`, and
`opposing_extents`. All twelve support calls shared a stable transport binding,
but the gate rejected the proposal: nine forward matches, three reverse
matches, seven `present`, five `nonmatch`, and no errors or indeterminate
outcomes. One positive missed `matched_geometry`, while two negatives were
false positives. Its phases stop at `proposal_frozen -> support_gate_rejected`,
with no query release or run archive. Event
`sha256:63983c4c918b23d8a009bca43a3390a1cf876bf96894521760761552dd8c11f8`
produced the intermediate thirteen-event ledger head
`sha256:65c8dd508f6c21e64b0c777a83159a470fbab12cfb8fee6adf588c0a9c400c8b`.

V9 also exposes an audit limitation. A support-rejected schema-v4 result has no
`run_archive`, while its outer public plan stores only the support-commitment
digest, not the nonce-bearing commitment preimage. Public `verify` therefore
cannot fully cold-bind and replay that immutable v9 file. This is deliberately
not described as v8-style fourteen-preimage verification.

Outer schema v5 closes the gap for new runs. Every exit now persists the exact
`support_commitment` object as well as its digest. For an ordinary
`support_rejected` result, cold verification binds that preimage to the public
plan, proposal receipt, canonically recompiled predicate, proposal freeze, all
twelve label-blind support receipts, verifier-side labels, gate counts, and
exact official PNG bytes. It requires the gate to reproduce a non-aligned
result and rejects any query observation or completed `run_archive`; no query
artifact is fabricated for a run that stopped before query release. Completed
and proposal-rejected schema-v4 files remain readable, while a schema-v4
support rejection is explicitly reported as lacking the required preimage.

Scientifically, v9 shows that vague visual correspondence needs a quantitative
symmetry and shape-matching leg; categorical self-judgment produced both a
positive miss and negative false positives. No result here estimates accuracy
on the 1,800-task official test split.

The schema-v5 v10 artifact,
[`bd_asymm_trap_bridge-trans_arc_cup_0000_v10.json`](runs/official_complete_drill_20260805/bd_asymm_trap_bridge-trans_arc_cup_0000_v10.json),
has file SHA-256
`0bdf82438b3b85b368f0c0fb93298f184fbae55b0b5777c06759670b53c3b8a7`.
It ended `proposal_error` because the sandbox could not resolve DNS before a
Codex response arrived. Its support-commitment preimage is nevertheless
present, but there is no validated structured response, proposal receipt,
support gate, query observation, or archive. Event
`sha256:dee13f7dae4e949882f516b8e8ca54eec7af8db0aa1fc47ca8a90aadb50195d7`
produced successor
`sha256:1a547a92e7897558e2f5f3e209545309d1f2ec41b4650d7b724ab4193840eff7`.
This is infrastructure failure, not evidence for either side.

V11,
[`bd_asymm_unbala_goldfish-asymmetric_crown_0000_v11.json`](runs/official_complete_drill_20260805/bd_asymm_unbala_goldfish-asymmetric_crown_0000_v11.json),
has file SHA-256
`0a324a7fc780dea392443a9afd54dbfe19fe5631d06ff1287abba7da342ac561`.
It proposed “Complete paired motifs with shared rotational handedness.” The
support gate rejected the claim with eight forward and four reverse matches:
ten `present`, two `nonmatch`, no errors, and no indeterminate outcomes. Public
verification successfully bound and reproduced all twelve exact official
support preimages; rejection is the verified result, not a missing archive.
Event
`sha256:395fbacc33c3bc206a581e2d85cf856b89e978ce6133a3a2574e193d6d7484ab`
produced successor
`sha256:8841cac62c203a2895a176c2cfbef8b97b46cfb33a6e0db2072c24efb54dc171`.

V12,
[`hd_closed_shape-has_obtuse_angle_0000_v12.json`](runs/official_complete_drill_20260805/hd_closed_shape-has_obtuse_angle_0000_v12.json),
has file SHA-256
`e7c62b4eb96e910d5ea2738fb6622ab9b469993befc7e0897906f5ed223960df`.
Its phrase was “A cyclic enclosure with an inward re-entrant feature.” It also
ended `support_rejected`: seven forward and five reverse matches, nine
`present`, three `nonmatch`, and no errors or indeterminate outcomes. Public
verification again reproduced all twelve exact support preimages. Event
`sha256:ce8f67fc54e3775932951c622d9f87dac805a12ac082bc66f5bc258764492c2e`
produced the current sixteen-event ledger head above.

Across this deliberately tiny v10–v12 smoke campaign, the proposer returned a
validated proposal in 2/3 attempts; 0/2 proposals passed the support gate; no
attempt released query pixels; and 0/3 completed. The two successful proposal
calls plus their 24 support observations produced 26 successful receipts:
233,921 known input tokens, 23,552 cached input tokens, 15,001 output tokens,
of which 10,694 were reasoning tokens. These are transport and gate-yield
statistics. Because no query was released, the campaign has no query accuracy.
The two verified rejections are useful diagnostics: verbose positive prose can
still collapse a conjunction or confuse different near-miss subgroups.

The canonical machine-readable aggregate is
[`official_complete_drill_smoke_v10_v12.json`](data/official_complete_drill_smoke_v10_v12.json),
with digest
`sha256:137536083875f40197d58363af5359750a10b385c1b0a5f1f9f2b11b882d3a66`.
`campaign_report.py` reproduces it from the three exact run records, checks the
exposure chain and unique receipts, and deliberately emits no score or accuracy
object when query release is zero.

## Historical material

The pre-rewrite macro-reuse control, action-program adapters, exploratory
visual pilots, and their original reports are preserved in Git at the
annotated tag `pre-bongard-complete-rewrite-20260805`. Stale working-tree copies
remain physically present pending explicit deletion, but they are excluded
from the canonical package and reproduction path. Their supplied atoms,
synthetic rerenders, partial subsets, and post-hoc diagnostics do not satisfy
the current official-corpus protocol.

See [`CONTINUATION_PLAN.md`](CONTINUATION_PLAN.md) for the current execution
roadmap. [`BONGARD.md`](BONGARD.md) is retained only as a compatibility pointer
to this page.
