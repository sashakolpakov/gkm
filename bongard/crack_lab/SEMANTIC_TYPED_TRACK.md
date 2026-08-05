# Typed Semantic Tracks

## 2026-08-05 falsification and corrected benchmark boundary

The prose-rubric experiment below is retained as a falsified exploratory
baseline, not as a valid generalization benchmark.  Its central object was
wrong: it evaluated

```text
VLM(panel, the whole candidate-rubric batch) -> membership
```

instead of first constructing candidate-independent evidence from the panel.
Consequently the proposal turn and scoring turns could silently choose
different objects, parts, contacts, and angular reference frames.  The
reported pair holdouts also refit only thresholds after the proposer had seen
all twelve labelled panels.  They are threshold-sensitivity probes, not
leave-one-out validation; for absolute rules the rotated error count is
mechanically six times the support error count and contains no new evidence.

The corrected `GROUNDED` path is:

```text
support pixels
  -> one headless-Codex vision turn selecting IDs from a closed measurement catalog
  -> candidate-independent registered witness extraction
  -> typed observations with intervals and provenance
  -> support-only synthesis of a positive Boolean predicate
  -> frozen predicate
  -> evaluation on hidden nuisance rerenders never shown to proposal or synthesis
```

Executable meaning lives in the registered witness/observable contracts, not
in the proposer's rationale.  The model may say "an asymmetric point contact"
and select the corresponding observable IDs, but it cannot author a detector,
choose a threshold, emit memberships, or reverse polarity.  All leaves in one
panel evaluation share a cached witness, so a conjunction cannot change its
segmentation or frame from term to term.  `SemanticAbsent` is a certified
negative fact; failed or ambiguous extraction is `Indeterminate` and blocks
admission.  Numeric comparisons use uncertainty intervals, and a boundary
overlap is also indeterminate.

This is still not `panel -> prose -> Lean proof`.  A proof assistant can prove
the typechecker/evaluator laws conditional on the registered extractors; it
cannot prove that pixels are bird-like without a formalized image ontology and
a verified extractor.  Open-vocabulary judgements therefore require a frozen,
content-addressed oracle protocol (for example contrastive prototypes and hard
negative foils) and remain `HYBRID`, never `PURE`.  An unconstrained absolute
VLM score in `[0,1]` is not a rigorous soft predicate and is not admitted.

This directory keeps the following Bongard paths separate in both artifacts
and claims.

- `UNRESTRICTED`: the existing `bongard_arena.py` + `bongard_legs.py`
  predicate-library path. `bongard-predicate-purity/v2` is a positive language:
  only the exact imports, restricted builtins, calls/values,
  methods/attributes, keyword forms, owned-scratch mutation, and bounded
  resource forms serialized by `predicate_capability_manifest()` are admitted;
  a listed module root grants no other API. Deterministic finite output is then
  audited and exact AST closures are priced under
  `bongard-predicate-pricing/v3` before bounded exhaustive rule selection.
- `SEMANTIC-PURE`: typed semantic cones. A candidate must compile through the
  witnesses required by its semantic claim before verifier or MDL selection.
- `GROUNDED`: headless vision selects only registered observable intents;
  candidate-independent witnesses and support-only Boolean synthesis produce a
  frozen predicate that is tested on hidden rerenders.
- `SEMANTIC-SOFT-EXPLORATORY`: the falsified opt-in prose-rubric experiment.
  One labelled joint Codex
  vision turn commits side-free operational rubrics; twelve fresh, label-blind
  single-panel turns score their visible cues; the harness then composes and
  verifies bounded membership values deterministically. It must never be
  reported as semantic-pure or as representation-level generalization.
- `HYBRID`: reserved for semantic cones plus explicitly priced residual guards.
  Hybrid results must not be reported as semantic-pure.

## Adaptation Notes

The repository already had the unrestricted control path:

- `bongard_arena.py`: rendering, predicate loading, MDL conjunction search,
  rotated pair holdout, and the exact `risk-then-cost/v2` selector.
- `bongard_legs.py`: proposer loop, promoted/WIP artifact discipline, taint
  checks, immutable paid-definition ledger, resume, and held-fixed no-share
  derivation.
- `predicate_pricing.py`: exact source-content identities and transitive AST
  closure/cost receipts for predicates, helpers, constants, and imports.
- `phase_d_protocol.py`: frozen corpus/control schemas, preregistration, track
  reports, and full-collection validation.
- `prepare_phase_d.py`: offline/write-once Phase D corpus preparation.
- `bongard_api_agent.py`: API proposer rung with paired semantic text and
  executable predicate code.

The typed semantic implementation is additive. It does not replace the
unrestricted path or promoted artifacts.

The current grounded implementation lives in:

- `grounded_predicate_ir.py`: closed typed Boolean IR, units, intervals,
  provenance, PURE/HYBRID taint, and four-valued observations;
- `grounded_observables.py`: the closed candidate-independent observable
  catalog and shared per-panel point-contact witness cache;
- `grounded_proposer.py`: one isolated headless-Codex support-image turn that
  may emit measurement intents only;
- `grounded_synthesis.py`: support-only interval-margin bounds and exhaustive
  positive-conjunction synthesis, with no polarity reversal;
- `run_grounded_semantic.py`: formula freeze before query creation, immutable
  artifacts, and model-free cold replay;
- `GROUNDED_BENCHMARK_20260805.md`: falsification evidence and the first live
  repaired/fresh-probe results.

The open-vocabulary exploratory implementation is deliberately separate:

- `hybrid_program_split.py`: deterministic content-distinct 6+6 support and
  6+6 query latent-program splits, with the same-template limitation recorded;
- `hybrid_claim_proposer.py`: exactly one side-free grammatical-affirmative
  claim and an exact support/prompt/schema/output Codex receipt binding;
- `grounded_contrastive_oracle.py`: a frozen three-pair reference bundle, two
  fresh fully swapped target presentations, secret role normalization, and an
  abstaining categorical decoder;
- `run_hybrid_contrastive.py`: query-latent commitment, formula freeze before
  query rendering, side-blind target order, HYBRID typed-IR evaluation, and
  evidence-only cold replay.

The first live `bird6` run is `UNSOLVED_HYBRID_EXPLORATORY`: `8/12` overall,
`9/12` determinate coverage, `8/9` determinate-only accuracy, three
indeterminates, and zero errors.  Its executable denotation is operational
resemblance to a frozen claim/reference bundle, not pixel-level truth of the
prose.  The same Bongard basic template underlies its content-distinct action
programs, so the run is a style/pose holdout rather than novel semantic-instance
generalization.  Full details and the artifact digest are in
`GROUNDED_BENCHMARK_20260805.md`.

Run a fresh one-problem grounded campaign and replay it with:

```bash
PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/run_grounded_semantic.py \
  --source basic --limit 1 --corpus-size 1 \
  --program-seed 20260805 --support-seed 20260805 --query-seed 20260806 \
  --model gpt-5.6-sol --reasoning-effort medium --minutes 15 \
  --out-dir bongard/crack_lab/semantic_grounded_runs/new-run

PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/run_grounded_semantic.py \
  --replay-artifact bongard/crack_lab/semantic_grounded_runs/new-run
```

## Main Invariant

The compiler must reject a candidate whose declared semantic content is not
supported by its typed dependency cone, registered proxies, or validated
gluings.

A candidate that says `triangle attached to square` must produce typed
evidence such as:

```text
ContourWitness -> PolygonWitness -> TriangleWitness
ContourWitness -> PolygonWitness -> QuadrilateralWitness
PartGraphWitness -> ContactWitness
```

It may not compile only through relative scalar metrics such as:

```text
bbox_aspect
bbox_occupancy
object_count
endpoint_count
cycle_count
```

Those measurements remain legal in the unrestricted track. In semantic-pure
they cover only the metric vocabulary their contracts actually name—for
example `aspect`, `occupancy`/`density`, `count`, `endpoint`, or `cycle`.
Relative thresholds on them cannot establish categorical adjectives such as
`thin`, `filled`, `connected`, `closed`, or `acyclic`. A categorical claim
needs the corresponding typed structural path; open/closed, for example, uses
an extracted `ContourWitness` followed by `contour_closedness`.

## SEMANTIC-SOFT-EXPLORATORY Claim Boundary

The deprecated exploratory soft flow is **not** panel -> verbal description -> Lean
predicate -> theorem proof. No Lean term is generated or checked. It is:

```text
12 direct panel images (6 labelled positive, then 6 labelled negative)
  -> one joint headless-Codex vision proposal (3-8 candidate concepts)
  -> frozen, content-addressed, side-free operational rubrics
     (claim + atomic visible cues + disqualifiers + fixed aggregation)
  -> 12 fresh Codex turns, each seeing only one neutrally named panel and all
     frozen rubrics (no label, neighbours, problem ID, selector policy, or
     feedback)
  -> SoftEvidence | SoftAbsent | SoftError for each panel and rubric
  -> harness-owned min/max/mean composition and disqualifier clamp
  -> strict >0.5 categorical cutoff (0.5 remains ambiguous), or an
     affirmative high-membership relative threshold
  -> support, 36 positive/negative pair-holdout folds, and polarity checks
  -> deterministic selection, persisted campaign, and cold artifact replay
```

The proposer analyzes the pixels directly; prose is the committed interface
between the joint visual analysis and the isolated panel scorers. It cannot be
rewritten after target scoring, and the soft path has no verifier-feedback
proposal rounds. Every admissible score is membership in the literal
affirmative claim, so only `high_positive` is legal. A concept such as `few
objects` or `absence of curves` must be stated and scored directly; selection
cannot rescue a bad concept by reversing its score.

`SEMANTIC-PURE` admits only claims that deterministic registered legs and
honest typed witnesses carry. `SEMANTIC-SOFT` may instead use open-vocabulary
visual judgements such as a decomposed `bird-like silhouette` rubric, but
calls them operational VLM evidence. A soft membership is bounded evidence
under that rubric, not a proof that the real-world category is true and not a
probability unless an independent calibration study establishes that
interpretation.
Open-world terms remain `MISSING_LEG` in the hard registry; in particular,
there is no default `bird-like` macro. The optional
`semantic_legs.soft_semantic_registry()` adds deterministic fuzzy operators
and analytic angle-obliqueness evidence, but does not weaken this gate.

What is mechanically enforced and replayable:

- exact schemas and content digests for frozen rubrics and evidence;
- one immutable image restaged outside the repository as `panel.png` per
  scorer turn, including through the generic scorer interface, with labels,
  neighbouring panels, problem ID, and feedback absent;
- only the visible scoring rubric and its rubric-only digest reach the scorer;
  hypothesis identity and threshold/comparison policy are excluded, while
  ordinal support-group language and panel-identity templates are rejected;
- finite membership in `[0,1]`, with abstention and failure represented as
  `SoftAbsent` and `SoftError`, never numeric sentinels;
- fixed unweighted `all`/`any`/`mean` composition, disqualifier handling,
  affirmative high-membership polarity, threshold fitting, pair holdouts, and
  selection;
- corpus manifests, canonical panel bundles, model-call receipts, atomic cue
  scores, and replay of every downstream decision without another model call;
- replay validation of the rubric and concept identities, exact cue identities
  and order, textual evidence, provenance, producer receipt, and membership
  recomputed from the frozen aggregation before numeric selection;
- strict cold loading of `campaign.json`, followed by exact recomputation of
  every candidate, selection, usage total, infrastructure flag, solved flag,
  and status, rebound to the frozen corpus and, when applicable, its
  shuffled-control manifest;
- separation of infrastructure failure from empirical failure: any scorer
  exception becomes typed `SoftError` evidence and
  `INVALID_SEMANTIC_SOFT`, never `UNSOLVED_SEMANTIC_SOFT`;
- for the typed soft extension, content-addressed prototype/calibrator
  contracts and analytic obliqueness whose uncertainty can only weaken its
  score.

What is not established by an accepted soft campaign:

- a formal proof of the VLM's open-world visual judgement;
- stability under re-querying the live model, or calibration as probability;
- representation-level generalization, because the proposer saw all twelve
  labelled panels before any pair holdout;
- performance on fresh unseen panels; that requires an independently frozen
  query corpus or external calibration suite;
- nuisance invariance: soft rubric morphisms are currently declarations in the
  artifact, not executed soft-verifier checks;
- a `SEMANTIC-PURE`-style conditional-MDL or paid-definition ledger: current
  soft selection uses the canonical rubric byte length, with no accounting for
  shared learned definitions across problems.

Run a one-problem headless Codex smoke campaign with a fresh output directory:

```bash
PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/run_soft_semantic.py \
  --source basic \
  --limit 1 \
  --corpus-size 1 \
  --model gpt-5.6-sol \
  --reasoning-effort medium \
  --minutes 15 \
  --scorer-workers 4 \
  --out-dir bongard/crack_lab/semantic_soft_runs/codex_eod_20260805
```

The runner cold-loads and validates the campaign it just wrote. Validate it
again in a fresh process, with no model calls, before reporting:

```bash
PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/run_soft_semantic.py \
  --replay-artifact \
  bongard/crack_lab/semantic_soft_runs/codex_eod_20260805
```

Increase `--limit` and `--corpus-size` together for a larger benchmark and use
a new output directory for every immutable campaign. Four scorer workers run
up to four of the twelve isolated single-panel turns concurrently; they never
batch panel images together. `run_semantic_cone.py --proposer codex` remains
the separate `SEMANTIC-PURE` runner; it does not activate the soft track.

## Generality Rule

Nothing problem-ID-specific is admissible in the harness. There is no
per-problem or composite-concept requirement table and no hard-coded composite
gluing. The harness does contain bounded operator/metric morphology and
registry-provided `proxy_for` vocabulary; novel composite claims, witness
demands, and gluings still come from the proposer and are verified
mechanically.

## Witness Honesty

A witness-producing leg must verify the structure it claims. `detect_contact`
returns a ContactWitness only when parts actually meet at a stroke junction;
`detect_intersection` requires a crossing (4+ incident branches); when the
relation is absent the leg raises `WitnessAbsent` instead of fabricating
evidence. Cone execution carries that typed absence; the verifier converts it
to a negative decision for a presence claim—or positive evidence for an
explicit witness-absence rule—instead of counting it as an implementation
crash.

The runtime has four dispositions, not a Boolean plus exceptions:
`Present`, `SemanticAbsent`, `Indeterminate`, and `Error`. `WitnessAbsent`
means the registered extractor established non-membership. By contrast,
`WitnessIndeterminate` means its evidence was insufficient or failed a
quality gate; it propagates as `IndeterminateValue`, produces a `None`
decision rather than `False`, counts as an error in support and every holdout,
and makes the cone inadmissible. Thus an unresolved negative panel can never
improve a predicate's accuracy. Ordinary exceptions remain implementation
errors. Every leg contract publishes disjoint `failure_modes` (semantic
absence) and `indeterminate_modes`, and raising an undeclared mode is itself
an implementation error. Both dispositions are included in the registry and
replay-policy fingerprints.

Each successful edge is also checked at runtime
against its declared codomain (including finite numeric measurements and
binary-panel constraints), so static diagram typing cannot be undermined by a
misbehaving implementation. Absence counts such as `contact_count`,
`intersection_count`, `part_count`, and `object_count` return 0 honestly.

## Files

- `visual_witnesses.py`: serializable witness dataclasses.
- `semantic_legs.py`: typed leg contracts and deterministic implementations
  (junction-based part decomposition, path-ordered contours, resampled corner
  detection, Taubin circle fitting), plus the explicitly selected
  `soft_semantic_registry()`. Placeholder implementations remain internal and
  are not registered for proposer use.
- `soft_semantics.py`: bounded typed results, content-addressed prototype and
  calibrator contracts, strict fuzzy operators, and analytic soft evidence.
- `semantic_soft_pipeline.py`: frozen prose-rubric proposal, isolated blind
  vision scoring, deterministic aggregation, verification, and replay.
- `run_soft_semantic.py`: immutable headless-Codex `SEMANTIC-SOFT` campaign
  runner. It is intentionally separate from `run_semantic_cone.py`.
- `test_soft_semantics_contracts.py` and
  `test_semantic_soft_pipeline.py`: typed-soft and information-boundary
  contract tests.
- `test_semantic_leg_contracts.py`: registry-wide constructive/codomain and
  failure-mode checks, plus completeness-checked invariance and equivariance
  matrices for every proposer-visible contract.
- `semantic_requirements.py`: general term-coverage audit. With no per-concept
  table, a declared term must be covered by witness types/legs in the score's
  dependency cone, by a used leg's own `proxy_for` contract, or by a
  declared gluing; otherwise MISSING_LEG with registry-derived suggestions.
- `semantic_compiler.py`: type checking, dependency checking, gluing
  validation, and `MISSING_LEG` enforcement.
- `semantic_verifier.py`: support and image-level LOO verification, executed
  cone-invariance (naturality) checks for declared morphisms, and per-panel
  gluing verification.
- `semantic_selection.py`: risk vectors, conditional complexity breakdowns,
  track labels, and Pareto frontier support.
- `cofibrations.py`: gluing-morphism contracts (see below) — machinery only,
  no concept-specific specs.
- `semantic_artifacts.py`: promoted/WIP artifact discipline, taint scan, and
  exact full-selection cold-replay gate.
- `run_semantic_cone.py`: verifier-in-the-loop runner (multi-round feedback,
  checkpoints, promotion/WIP snapshots).

## Adding a Witness

1. Add a dataclass to `visual_witnesses.py`.
2. Include geometry, confidence/residual, source IDs, child witnesses, and
   provenance.
3. Keep it trace-serializable via `to_trace`.

## Adding a Leg

1. Implement a deterministic function in `semantic_legs.py`.
2. Register it with a `LegContract` containing domain, codomain, the correct
   distinction between scalar invariances and witness equivariances, failure
   modes, complexity cost, and version.
3. Add constructive, adversarial-negative, and declared-transform tests before
   exposing the leg to the proposer.

## Term Coverage (no per-concept lexicon)

There is no requirement table to edit. When a hypothesis declares a term,
the compiler audits it mechanically:

1. controlled tokens of the term are matched against dependency-cone witness
   types, leg names, and contract aliases with registry-driven normalization
   and bounded stem/prefix/suffix heuristics; whole-name arbitrary fragments
   are not evidence;
2. tokens explicitly named by a used leg's `proxy_for` contract are covered
   (e.g. `bbox_occupancy` declares only `occupancy`/`density`);
3. tokens carried by a validated gluing's typed endpoints and executed
   attachment leg are covered; its proposer-written name is never evidence;
4. cardinality, comparison, bounds and negation are parsed into a clause-bound
   score operator; absolute operators execute as fixed predicates, while only
   relative operators use a fitted threshold;
5. anything else raises MISSING_LEG with suggestions computed from the
   registry, never from a concept table.

The same canonical claim representation is used for prose/structured-header
agreement and verifier execution. It retains metric identity, counted-head
identity, polarity, and clause-local operator scope, so swapped cardinals,
exact-versus-at-most changes, nested negation, and length/residual or
aspect/occupancy substitutions fail before empirical fit. Unsupported
distributive/exclusive quantifiers fail rather than collapsing to an aggregate.
Numeric input is bounded; unsupported signs, decimals, exponent notation, or
symbols are rejected rather than punctuation-stripped into a different claim.

To make a new term expressible, add a general leg/witness whose contract
carries that vocabulary — not a mapping entry.

## Cofibrations are Gluings, and Proposer-Generated

A cofibration declaration is treated operationally as an
`A -> A ⊔_I P`-shaped gluing claim. The checker does not construct a
categorical pushout or verify its universal property. It requires nonempty
declared interface/patch fields, a structurally glue-equivalent embedded or
projected source up to ID renaming and numeric tolerance, and an executed
attachment relation bound to the mapped source identity.

Specs are NOT hard-coded per concept in the library. The proposer emits them
inside its cone IR (`cofibrations` field), binding `source_node` and
`target_node` to diagram nodes. A target-to-source projection must be executed
when the source is not otherwise load-bearing, and the attachment leg must
consume the target, produce a typed relation witness, and lie on the final
score path. The relation endpoints must contain the glued source identity after
the declared glue map, so an unrelated attachment elsewhere in the target
cannot discharge the claim. Interface and added fields are nonempty, disjoint,
structural target fields; bookkeeping such as confidence, residual, IDs, and
provenance cannot stand in for an interface or patch. The verifier runs
`verify_cofibration` on positive-panel witness values; negative panels may
honestly lack the claimed gluing. Hard-coded specs are allowed only as unit-test
fixtures. `select_largest_part` supplies the first generic
PartGraphWitness-to-PartWitness projection, and end-to-end fixtures cover both
a valid attachment and an unrelated-contact rejection.

## Replay and Promotion

A promoted semantic cone is not certified in isolation. Its
data-self-contained RunSpec contains the canonical panels and the complete
candidate set accumulated across proposer
rounds, each candidate's round/index/count, cone and verification digests, expected
verification and conditional-complexity evaluation, the selected candidate ID,
exact verifier policy, registry and source fingerprints, and a fingerprint of
the runner/selector module. The checkpoint manifest and the entire promoted
selection record—not just its ID—must match this evidence. A fresh Python
process under the matching recorded environment re-verifies every candidate,
recomputes all evaluations and selection, and requires the resulting winner's
cone digest to equal the promoted payload. Promotion also independently
requires zero support, LOO, pair-LOO, predicate, naturality, and cofibration
errors, with no unchecked
morphisms or semantic issues. Fixed semantic predicates are reused as-is;
only explicitly relative scalar rules fit a threshold inside each LOO fold.

## Cone Invariance is Executed

Declared `preservation_morphisms` are not decoration. Supported actions are
attempted on every panel. Translations, reflections, and named 90°/180°/270°
rotations use exact array actions; generic `rotate` also includes thresholded
off-grid probes. Unsupported actions, or actions inapplicable to any panel,
are reported in `unchecked_morphisms`; otherwise decision drift increments
`naturality_errors` and rejects the cone.

A broader rotation/reflection/translation battery is reported separately as
`stress_errors`. It is deliberately not an admission rule: reflection is not
label-preserving for every possible concept (chirality is the obvious
counterexample), so the harness may not invent that declaration on the
proposer's behalf.

Leg-level transform metadata is audited separately from whole-cone
naturality. Scalar measurements declare invariances when their values should
remain fixed; coordinate-bearing witnesses declare equivariances when their
geometry should be transported by the action. The contract laboratory derives
its matrix from the registry itself, so a newly advertised transform cannot
silently escape a direct check. Raster extractors currently advertise only
transformations supported by their sampling behavior; unsupported scale claims
are not smuggled in as invariances.

Separate reachability tests exercise 648 full raster compositions over angles
0–85° in 5° steps and fixed-geometry stroke widths 1/3/5/7. They cover
line/arc/S-curve inflections and parts, triangle/quadrilateral classification,
circle fitting, T-junction attachment, four-way intersection, and three-ray
radial detection. These composition tests are stricter than the individual
leg metadata matrix: they catch contradictions such as a contour fitting an
honest straight-line witness while a downstream counter invents dozens of
inflections.

## Example MISSING_LEG

For a hypothesis saying `two intersecting circles` but using only
`bbox_occupancy`, the compiler's coverage audit reports a structured failure
with registry-derived suggestions:

```text
MISSING_LEG
semantic term: two circles
required: CircleIntersectionWitness + CirclePairWitness + CircleWitness
available paths terminate at:
- Measurement
- Object
- Panel
- Scene
missing:
- circle_pair_intersection
- circle_residual
- fit_circle
- fit_multiple_circles
```

This is a useful result. It means the system identified representation
poverty instead of accepting a short proxy.

## Current Expressible Structure

Faithful typed paths now exist for:

- polygon side counts via path-ordered, arc-length-resampled contour fitting
  (triangle/quadrilateral classification refuses wrong side counts)
- line segments with residual rejection, and circles/arcs via normalized
  Taubin fitting (an open arc honestly refuses `fit_circle`, while `fit_arc`
  refuses closed, degenerate, reversing, undersupported, or high-residual data)
- open/closed via contour topology (`contour_closedness`)
- part decomposition at real stroke junctions, with honest
  contact/attachment (3-branch junctions) and intersection (4+ branches)
- local cross-part junction angles via scale-relative incident-ray fits,
  preserving raw angle magnitude and typed fit/absence failures
- radial arrangement with measured angular/radial uniformity
- skeleton endpoint/branch/cycle counts
- exact two-circle intersection construction: the returned one or two points
  lie on both input circles, while disjoint, contained, concentric, or
  coincident pairs refuse the witness
- explicitly metric scalar paths such as `bbox_occupancy`; categorical aliases
  such as `bbox_fill`, and the misleading geometric-area name
  `largest_area`, remain internal

The circle-intersection path now reaches through a merged raster component.
`fit_multiple_circles` uses Hough geometry only to propose candidates, then
Taubin-refines them against the actual ink and requires full angular support,
bounded RMS/q95/max radial error, distinct circles, at least 90% joint ink
coverage, and balanced assignment. A constructive test parses one merged pair
of intersecting circle strokes, recovers both circles, and feeds them to the
real intersection constructor; square/cross distractors and poor joint fits
remain negative controls.

Composite names (any concept with no matching registry structure) must be
assembled from primitive witnesses plus proposer-declared gluings. There is
no default black-box concept leg and no per-concept harness table.
Compatibility helpers that overstate their evidence—tangent transport
without tangent vectors, curvature/containment/symmetry relabelings,
shared-point/tangency relabelings, and polygon fitting presented as explicit
line decomposition—remain quarantined from the proposer-visible registry.

## Runner Discipline

`run_semantic_cone.py` gives the proposer up to `--rounds`
verifier-in-the-loop turns per problem (structured tool output, so malformed
JSON cannot kill a run). Each round's compile errors, MISSING_LEG
structures, per-panel score tables, misclassified panels and invariance
violations are fed back mechanically.

The semantic Anthropic proposer resolves its model alias to a concrete provider
ID and requires every Messages response to report that exact ID; omission or
substitution fails before proposal parsing. This is separate from the
unrestricted Codex CLI contract. That path binds the exact task, current and
proposed source/log, raw and semantic panel identities, structured output,
unique thread/event identity, explicit model/reasoning request, positive usage,
and pinned CLI/launcher. It represents the JSONL model field as unreported when
the stream omits that optional evidence. Its consuming attempts carry
persisted receipts.

Mirroring the unrestricted track:

- exact 12-panel semantic fits are promoted into
  `agent_solutions/<tag>_semantic/` (`checkpoint.json`,
  `promoted_cones.json`, harness-only `results.json`, `replay_specs/`,
  `replay_receipts/`, `README.md`), gated on a taint scan and a fresh-process
  replay of every promoted RunSpec;
- failed attempts are snapshotted append-only under
  `wip_context/<opaque_id>/<timestamp>/` and never admitted;
- ground-truth concept names never enter the run workspace;
- each RunSpec embeds canonical panel bytes and cone IR, the complete
  candidate/selection evidence and manifest, plus panel/cone digests, verifier
  policy, registry and related-source fingerprints, Python, and dependency
  versions; promotion refuses any digest, verdict, or selection mismatch.

## Phase D Corpus and Controls

Phase D preparation is implemented. A paid unrestricted-only n=1 exploratory
pilot completed on 5 August 2026, but the full externally committed default
study—including this semantic-pure track—has not been run.
The preparation command samples one declared maximum and freezes both the
manifest and exact canonical panel bytes. When `source=both`, basic and
abstract use independent source RNG streams and are then combined by the fixed
four-basic/one-abstract interleave. The active n=1, n=5, and n=25 datasets are
nested prefixes of that one ordered maximum; neither runner may resample at a
smaller limit to create a scale.
The local write-once preregistration is a reproducibility manifest, not an
external timestamp. A confirmatory claim requires its digest to be published or
externally committed before the first proposer call.

The default preregistration has 27 arms. At each of the three scales,
`UNRESTRICTED` and `SEMANTIC-PURE` each have a primary arm and three
shuffled-side replicates, and `UNRESTRICTED` alone has no-share. A shuffled
replicate deterministically places three panels from each original side on
each controlled side, then runs the entire adaptive pipeline with the
controlled labels. It is not a post-hoc evaluation of a primary model.

`SEMANTIC-SOFT` is not part of those 27 preregistered Phase D arms. Its current
campaign runner is exploratory and writes a separate `SEMANTIC-SOFT` artifact;
its outcomes must not be inserted into a `SEMANTIC-PURE` report. Adding it to a
confirmatory study requires a new preregistration and independent hidden-query
or calibration evidence matching the claim being made.

No-share has a different interpretation. It is derived offline from the
corresponding unrestricted primary trace, copies its exact accepted predicate
sources, rules, risks, and outcomes, and repays the full reachable definition
cost for every accepted-rule use. It measures the accounting effect of sharing
while holding definition availability and selected behavior fixed. It is not
an independently proposed arm and cannot support a causal solve-rate claim.
`SEMANTIC-PURE` has no no-share arm until its registry distinguishes learned
legs from fixed base machinery.

Per-arm reports carry the exact track, condition, label policy, sharing policy,
corpus digest, source-trace digest, and—only for no-share—the parent primary
trace. `validate_complete_report_collection` requires every preregistered arm,
checks that scale records are nested prefixes, and rejects a no-share report
whose primary outcomes, rules, or source evidence differ. Checkpoints and
promotion additionally bind the embedded panel bundle and shuffled assignment
where applicable. Unrestricted resume/promotion sequentially re-executes every
terminal attempted source on those embedded panels and compares selected/fold
rules, risks, errors, and exact pricing receipts under a fingerprint of the
arena, pricer, policy, runtime, and numerical dependencies. Semantic promotion
uses the separate fresh-process full-candidate replay described above.

The sequential workflow is:

1. Run `PYTHONHASHSEED=0 prepare_phase_d.py --source both --limit-per-source 25
   --scales 1,5,25` once; it performs no model/API work.
2. Supply its `phase_d_preregistration.json` and one exact `--arm-id` before
   either runner constructs a proposer. Supply that arm's exact generated
   `execution_tag` as `--tag`; for semantic runs use the canonical
   `semantic_runs/<execution_tag>` output directory.
3. Keep the maximum sampler limit fixed. Extend a semantic arm with
   `--corpus-size 1`, then `5`, then `25`; extend an unrestricted arm with
   `--max-problems=1`, then `5`, then `25`. For every primary or shuffled
   proposer family, n1 is the only legal fresh start. n5 requires a complete,
   replay-valid n1 checkpoint in the same execution-tag artifact, and n25
   requires n5. Same-scale resume is allowed; a fresh higher-scale start,
   shrink, skipped predecessor, or incomplete predecessor is rejected before
   proposer construction and before any write. Each scale still publishes its
   own arm report. No-share scales remain independent derived artifacts rather
   than a resumable proposer family.
4. Run each shuffled replicate separately with its recorded control seed and
   replicate. Derive unrestricted no-share only after its parent primary trace,
   using `bongard_legs.py --no-share-from=<primary-execution-tag>`. Give the
   n1, n5, and n25 derivations their preregistered scale-specific execution
   tags and matching
   `--max-problems=` values; no-share is not a resumable proposer run.
5. Under the same `PYTHONHASHSEED=0`, pass every explicit execution-artifact
   `track_reports` directory to `collect_phase_d.py` with the frozen
   preregistration and a write-once output path. Treat the reports as a study
   only after it publishes the canonical campaign with one source/checkpoint/
   results/replay certification per arm and all track-specific replay gates
   accept them.

The prepared manifest also freezes exact model/ladder and time/token limits,
semantic rounds/error thresholds/lambda, selector constants, the complete
`bongard-predicate-purity/v2` capability manifest,
`bongard-predicate-pricing/v3`, harness source hashes, Python and its hash-secret
probes, dependency versions, and the exact Codex CLI version and launcher
identity. Production unrestricted records carry the causal input/output and
panel chain, unique turn identity, explicit requested model and reasoning
effort, positive usage, and honest provider-model evidence status; semantic Messages responses independently pass
the concrete provider-model equality gate above. Code or runtime drift requires
a new preregistration rather than an attempted resume. An unset or `random`
`PYTHONHASHSEED` cannot support a later-process Phase D replay and is therefore
rejected by the preparer.

The proposer sees all 12 labeled panels. For relative score claims,
`loo_errors` and `rotated_loo_errors` measure scalar-threshold refitting after
the representation was selected. Absolute/count/binary/witness claims instead
reuse their fixed executable predicate in every fold. Neither is an untouched
representation-level holdout. Console/checkpoint labels retain threshold-LOO
terminology for the relative case. Exact status requires zero support errors
and zero errors in both diagnostics. Broad stress drift qualifies the status as
`SOLVED_SEMANTIC_PURE_STRESS_FLAGGED` rather than disappearing.

For the unrestricted path, this all-panels exposure also limits what rotated
LOO can establish. Panel-identity lookup is instructed against and its payload
is charged/visible, but the rotation cannot categorically distinguish
memorization of the displayed panels from a reusable representation. Any
generalization claim needs separate unseen-instance evidence.

## Kolmogorov Selection

Verifier acceptance is the selection gate. A `CandidateEvaluation` is recorded
for every verification, but only candidates whose `verification.accepted` is
true enter ranking. Under the live exact policy this includes exact support,
LOO, and rotated-LOO requirements as well as semantic admissibility. Among
accepted implementations, selection uses risk plus conditional complexity:

```text
F = R + lambda C(M | L)
```

The runner selects the accepted Pareto frontier by the measured coordinates
`R_support`, `R_rotated_LOO`, `R_naturality`, and `R_parser_stability`, plus a
harness-derived `ComplexityBreakdown`. The latter stress coordinate records
whole-cone decision drift under the broad dataset battery. `R_contrast`,
`R_counterfactual`, and `R_archive_regression` are unimplemented and remain
explicitly unmeasured (`null`), so the recorded score is named
`conditional_free_energy`; the full-vector `free_energy` remains null.
