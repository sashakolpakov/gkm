# Bongard Concept-Induction Experiments

This subject studies reusable abstraction in Bongard-style concept induction. The
full subject guide is [`BONGARD.md`](BONGARD.md); this page also documents the newer
typed semantic-cone track. The evidence-backed continuation order is recorded in
[`CONTINUATION_PLAN.md`](CONTINUATION_PLAN.md).

The controlled symbolic experiments and the rendered Bongard-LOGO adapter have
different evidential scope. The former measure whether a shared predicate macro is
selected over duplicated rule bodies when primitive atoms are supplied. They do not
show discovery of those atoms from pixels. The semantic-cone track below adds typed
factorization and explicit witness checks; unrestricted predicate search remains a
separate control path.

The unrestricted `predicates.py` crack loop remains the control path. Its
versioned positive capability language, `bongard-predicate-purity/v2`, admits
only the exact imports, builtin names, module calls/values, instance
methods/attributes, keyword forms, locally owned scratch mutation, and bounded
resource forms serialized by `predicate_capability_manifest()`; listing a
module root does not authorize any other API below it. Execution receives the
restricted builtin set plus the import hook needed for AST-approved imports,
not Python's ambient builtin namespace. The authoritative verifier also
rejects import-time execution, I/O/dynamic namespace access, exceptions,
non-finite or schedule-dependent outputs, then prices the exact source snapshot
under `bongard-predicate-pricing/v3` before composing and admitting it. The
semantic-pure path is
stricter: a proposer supplies typed semantic cones, and the harness
type-checks, validates semantic witness coverage, and verifies them separately.
The live runner now uses an explicit partial-risk conditional-MDL selector and
refuses promotion until a data-self-contained RunSpec replays exactly in a
fresh Python process under its recorded matching code/runtime environment.
Replay re-verifies every candidate seen across proposal rounds,
recomputes its harness-derived evaluation, reruns the fingerprinted selector,
and checks that the reproduced winner is the promoted cone. Unimplemented
contrast, counterfactual, and archive-regression risks remain serialized as
`null`; this is not a claim that those risks are zero.

Each unrestricted production turn uses non-interactive `codex exec`, explicitly
requesting `gpt-5.6-sol` and medium reasoning. Its prompt and private,
mode-0700 working directory supply twelve copied PNGs and the current
library/log, but no harness-workspace path. The run
is ephemeral and schema-constrained, ignores user config and rules, uses a
read-only sandbox, and disables shell/unified exec, search, apps, plugins,
browser/computer use, hooks, skills, and sub-agents. Only a complete structured
source/log response is applied afterward. JSONL receipts bind positive usage,
unique thread/event identity, the exact task and current/proposed source/log,
raw PNG view and semantic panel set, structured output, schema, CLI version,
and resolved launcher-file identity; any tool/file/web/MCP event is rejected.
This is locally constructed causal provenance, not provider-signed attestation,
and the launcher digest does not cover every transitive package/native/runtime
byte. The documented JSONL stream may omit its model field, so receipts
distinguish an explicit `--model` request from JSONL-reported model evidence
instead of fabricating `actual=requested`.
With the currently pinned Codex CLI 0.146.0, strict config rejects the newer
`tools.view_image=false` key. The runner therefore withholds repository paths,
uses an outside-repository image-only view, and rejects any emitted
`view_image` event, but cannot claim that this built-in local-image surface was
removed before execution. A CLI change requires a new preregistration.
Separately, the semantic Messages
proposer resolves a model alias to a concrete provider ID and rejects every
response that omits or changes that ID.

## Architecture

```text
panels
  -> proposer
  -> typed cone proposals
  -> leg registry
  -> mechanical compiler
  -> semantic admissibility gate
  -> exact support / fixed predicates or threshold-refit diagnostics
  -> declared naturality / gluings
  -> separate dataset-wide transform stress diagnostics
  -> partial-risk conditional free-energy selection
  -> data-self-contained candidate-set RunSpec + selection replay
  -> promoted artifact + replay receipt
```

The proposer is "cofibered" in the operational sense: for each problem object
`p`, it proposes candidates in the fiber of semantic cones over `p`. In
semantic-pure mode it does not write final image-processing code. New reusable
legs are requested as typed missing arrows and are admitted only after
separate implementation, contract checks, replay, and pricing.

The compiler rejects undeclared or unrecognized prose claims, degenerate gluings,
proposal-controlled complexity, invalid call parameters, and unexecuted
contrast declarations. Expected absence of a typed witness on a negative panel
propagates to a negative decision rather than being counted as a predicate
crash, but only when its machine-readable failure mode is advertised by the
executing leg. Successful values are checked against their runtime codomains,
so non-finite measurements, wrong witness classes, and non-binary
`BinaryPanel` values fail closed. Node and leg dependency identities are kept
separate, so a decorative witness cannot become load-bearing through a name
collision. Placeholder legs that merely relabeled contact, symmetry,
containment, tangent/curvature data, or line decomposition are not
proposer-visible.

Quantitative language is compiled too. Comparators, negation, counted heads,
and shared bounds are bound to their own clause and metric; unsupported
quantifiers and nested negations fail closed; bounded numeric syntax fails
closed; and every
calibrated term must execute through the final measurement (or an exact typed
pair construction). Counts, binary aliases, numeric bounds, exact equality,
conjunctions, and witness presence/absence use fixed predicates throughout
support, LOO, naturality, and stress. Only explicit relative comparisons use a
learned scalar threshold. This prevents metric swaps such as proving line
length with residual or aspect ratio with occupancy.

Phase C now includes a registry-derived primitive contract laboratory. It
executes every current proposer-visible leg on a constructive fixture, probes
every advertised failure mode, and derives complete matrices for declared
scalar invariances and coordinate-bearing witness equivariances. Scalar proxy
vocabulary is deliberately bounded: continuous relative proxies are
metric-specific, while count contracts additionally name the entities they
count. `bbox_occupancy` may support an occupancy/density comparison, but it
cannot launder the categorical claim `filled`; neither continuous nor count
proxies can establish unrelated claims such as `thin`, `connected`, `closed`,
or `acyclic`.

The laboratory also executes 648 raster-composition configurations over
angles 0–85 degrees in 5-degree steps and stroke widths 1/3/5/7. Straight,
arc, and S-curve paths retain coherent inflection/part counts; acute triangles
and quadrilaterals remain reachable through polygon classification; circles
retain their fit; and T-junction, four-way-cross, and three-ray paths retain
their part/contact/intersection/radial topology. This phase remains open for
the same breadth on specialized composition paths and for corpus-level stress.

Circle intersection illustrates that boundary. Given a valid
`CirclePairWitness`, `circle_pair_intersection` constructs one or two points
that lie on both circles and refuses absent/malformed geometry. The tested
upstream pixel path now handles the harder merged case: Hough candidates from
one connected component are Taubin-refined, checked for full angular support
and radial tails, then pair-checked for distinctness, joint ink coverage, and
balanced support. End-to-end tests recover two intersecting raster circles and
construct their real intersection points. This primitive result alone is not
an exact corpus solve.

`SOLVED_SEMANTIC_PURE` is deliberately scoped to the displayed 12-panel
instance: the proposer sees all labeled panels. For a relative claim the LOO
metrics refit only its scalar threshold after representation selection; for an
absolute claim they execute the same fixed predicate without refitting. They
are diagnostics, not an untouched estimate of semantic generalization. A
transform-stress failure receives the qualified
`SOLVED_SEMANTIC_PURE_STRESS_FLAGGED` status.

## Phase D: Protocol Ready, Exploratory n=1 Pilot Complete

The engineering gate is implemented, and an unrestricted-only paid protocol
pilot completed on 5 August 2026. This was a deliberately tiny three-arm
execution: one Basic problem, one primary arm, one shuffled-side replicate,
and the held-fixed no-share derivation. It is not the default 27-arm study and
is not confirmatory: its local preregistration digest was not externally
committed before the first proposer call, it contains no semantic-pure arm,
and n=1 supports no solve-rate or reuse conclusion.

The write-once campaign is
[`crack_lab/phase_d_runs/codex_eod_20260805_v3/campaign.json`](crack_lab/phase_d_runs/codex_eod_20260805_v3/campaign.json),
with campaign digest
`sha256:8be70918d2b57811a66787cdff845dbcb445eaf8e073f61443cea698845dfcf2`.
All three artifact certifications and terminal cold replays regenerated
exactly. The primary was an ordinary miss (0/1; held-out 0.500, train 0.583),
the shuffled arm ended in one canonical verifier failure (0/1, zero
admission), and no-share inherited the primary miss (0/1). Six unique
headless Codex turns requested `gpt-5.6-sol` at medium effort and receipted
64,641 input plus 28,489 output tokens; one earlier shuffled call timed out as
infrastructure before producing a consuming receipt. No candidate source was
admitted. See the [pilot record](crack_lab/phase_d_runs/codex_eod_20260805_v3/README.md)
for the exact boundary. The historical two-problem Sonnet smoke predates this
protocol and still does not count as current Phase D evidence.

`prepare_phase_d.py` samples one maximum corpus and freezes both a manifest and
the exact panel bytes. With `--source both`, basic and abstract sampling use
independent source RNG streams before the fixed four-basic/one-abstract
interleave, so adding one source cannot perturb the other. Scale is then an
ordered prefix of that one frozen maximum: the intended sequence is 1, then 5,
then 25 problems, always with the same maximum-corpus digest. A smaller scale
must never be produced by resampling with a smaller `--limit`.

Each shuffled-side replicate is a full adaptive control, not a post-hoc score
permutation. For every problem it deterministically assigns three panels from
each original side to each controlled side, records the assignment, and then
runs the same proposer, verifier, selector, replay, and promotion path. The
default preregistration has 27 separately labeled arms: at each of three
scales, both tracks have one primary and three shuffled-side arms, while only
`UNRESTRICTED` has a no-share arm. `validate_complete_report_collection`
requires every arm, checks that the 1/5/25 records are nested prefixes, and
cross-checks unrestricted no-share records against their primary parents.

Unrestricted rules use exact AST-backed pricing. A selected `p_*` atom pays
for the transitive closure of its predicate, helpers, constants, and imports,
identified by exact source content; shared nodes inside one rule are counted
once. Non-comment LOC, literal/call payload, and executable AST structure
price definitions, so deeply nested or semicolon-packed one-line logic is not
nearly free. Every rule use also pays its call/binding structure cost, so
moving a conjunction behind one predicate name does not make its
implementation free. The selector
exhaustively compares constants and conjunctions of up to two atoms from the
bounded 24-atom candidate set, minimizing empirical risk first and conditional
cost second (`risk-then-cost/v2`). Only definition identities from previously
accepted rules enter the immutable paid ledger; unused library code receives
no discount.

Panel-identity lookup is instructed against and its literal/call payload is
charged and remains visible in the persisted source and pricing receipts.
Rotated leave-one-out cannot,
however, categorically distinguish such memorization: the proposer has already
seen all 12 displayed panels, so an identity-keyed table can classify a held-out
panel. Claims of representation-level generalization therefore require
separate unseen-instance evidence.

The unrestricted no-share arm is deliberately held-fixed accounting. It
copies the primary arm's accepted sources, rules, risks, and solve outcomes,
then charges the full transitive definition closure on every accepted-rule
use. It is useful for measuring the accounting benefit of reuse, but it is not
an independent proposer run and supports no causal solve-rate claim. There is
no semantic-pure no-share arm: that comparison is undefined until the semantic
registry has an explicit learned/base split.

Both runners bind checkpoints and promoted artifacts to the frozen corpus,
embedded panel bundle, condition/control identity, and ordered record prefix.
Semantic promotion cold-replays every candidate and the selected winner in a
fresh process. Unrestricted resume and promotion replay every terminal record
in source order from its exact attempted predicate snapshot and embedded
panels, rerun priced selection, and compare rules, folds, risks, errors, and
receipts under a fingerprint of the arena, pricer, policy constants, Python,
and numerical dependencies. A `VERIFIER_FAILURE_UNRESTRICTED` record is a
distinct canonical zero-admission sentinel; its exact failure must reproduce
on fresh replay and is counted separately from ordinary unsolved records. The
final `predicates.py` must equal the last
accepted source. Resume extends the same prefix; it must not silently change
corpus, condition, sharing policy, runtime fingerprint, or artifact identity.

## Current Entry Points

- `crack_lab/prepare_phase_d.py` - offline maximum-corpus, control, and
  preregistration preparation; it constructs no proposer and makes no API call.
- `crack_lab/phase_d_protocol.py` - frozen corpus/control schemas, 27-arm
  preregistration, per-arm reports, and complete-collection validation.
- `crack_lab/collect_phase_d.py` - offline exact-arm artifact certifier and
  atomic, write-once canonical campaign publication.
- `crack_lab/bongard_legs.py` - unrestricted adaptive runner and held-fixed
  no-share derivation.
- `crack_lab/codex_proposer.py` - hardened ephemeral Codex structured transport
  and model-usage receipt parser.
- `crack_lab/bongard_arena.py` - bounded exhaustive rule verifier/selector.
- `crack_lab/predicate_pricing.py` - exact transitive AST definition pricing.
- `crack_lab/run_semantic_cone.py` - semantic-cone experiment runner.
- `crack_lab/cofibered_proposer.py` - LLM-backed cone proposal interface.
- `crack_lab/semantic_ir.py` - JSON-serializable typed IR.
- `crack_lab/semantic_compiler.py` - factorization-enforcing compiler.
- `crack_lab/semantic_verifier.py` - support and leave-one-out verifier.
- `crack_lab/semantic_legs.py` - initial typed visual leg registry.
- `crack_lab/test_semantic_leg_contracts.py` - registry-wide primitive contract
  and invariance/equivariance laboratory.
- `crack_lab/semantic_selection.py` - explicit risks and conditional complexity.
- `crack_lab/semantic_replay.py` - canonical panels/cones and provenance RunSpec.
- `crack_lab/replay_semantic_runspec.py` - fresh-process replay entry point.

## Phase D Prefix Workflow

Preparation is offline and write-once:

```bash
PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/prepare_phase_d.py \
  --source both --limit-per-source 25 --scales 1,5,25 \
  --out-dir bongard/crack_lab/phase_d_runs/preregistered
```

This produces `corpus_manifest.json`, `corpus_panels.json`, three frozen
shuffled-control manifests, and `phase_d_preregistration.json`. Before any
paid proposer is started, pass both `--preregistration` and the exact
`--arm-id` to the chosen runner. Every arm also contains its generated
`execution_tag`; pass that exact value as `--tag`. Primary and shuffled arms
at n=1/5/25 share one family tag so their checkpoint can grow, while every
no-share scale has its own immutable tag. For semantic families, the working
directory is `crack_lab/semantic_runs/<execution_tag>`; preregistered runs
reject an arbitrary scratch directory. For semantic runs, keep `--limit 25`
fixed and advance `--corpus-size` through `1`, `5`, and `25`; for unrestricted
runs, keep `--limit=25` fixed and advance `--max-problems=1`, `5`, and `25`.
For every primary or shuffled proposer family, n1 is the only legal fresh
start. An n5 invocation requires a complete, replay-valid n1 checkpoint in the
same execution-tag artifact, and n25 analogously requires n5. Same-scale resume
is allowed; a fresh higher-scale start, shrink, skipped scale, or incomplete
predecessor fails before proposer construction and before any write. Each scale
still publishes its distinct arm report. No-share scales are independent
derived prefix artifacts, not resumable proposer families. Use
`--condition shuffled-sides` plus the preregistered replicate for shuffled
arms. After an unrestricted primary prefix exists, derive its accounting-only
control with
`bongard_legs.py --no-share-from=<primary-execution-tag>` and the matching
no-share arm ID. Use the exact scale-specific no-share `execution_tag` and the
matching `--max-problems=` value; these are prefix replays, not a resumable
proposer family. Run preparation, every runner invocation, and collection
under the same explicit `PYTHONHASHSEED=0`, interpreter, dependencies, and
Codex CLI version/launcher fingerprint. Report files are emitted per arm. Finalize only from
explicit execution-artifact `track_reports` directories (which are scanned
for direct JSON children, never recursively):

```bash
PYTHONHASHSEED=0 .venv/bin/python bongard/crack_lab/collect_phase_d.py \
  --preregistration bongard/crack_lab/phase_d_runs/preregistered/phase_d_preregistration.json \
  --report-dir <track-reports-dir> [--report-dir <another-track-reports-dir>] \
  --out bongard/crack_lab/phase_d_runs/phase_d_campaign.json
```

The collector requires every preregistered arm exactly once and requires each
report to occupy the artifact named by its `execution_tag`. It cold-replays
the originating unrestricted checkpoint or semantic RunSpecs, reconciles the
checkpoint, results, source/cones, receipts, and report prefix, runs the full
cross-arm validator, and embeds an ordered certification for every arm. It
then atomically publishes a deterministic `bongard.phase-d-campaign/v6`
document. The output path is write-once: an identical rerun is idempotent,
while a different, corrupt, misplaced, or symlinked result fails closed.

Preparation freezes more than the arm table: it binds the exact proposer
ladder/limits, selector policy, semantic rounds/tokens/error bounds/lambda,
harness source hashes, the complete `bongard-predicate-purity/v2` capability
manifest and `bongard-predicate-pricing/v3` policy, Python runtime and hash
probes, scientific dependency versions, and the exact Codex CLI version and
resolved launcher digest used by the unrestricted track. Unrestricted
production attempts additionally bind the requested model/effort, any model
identity actually reported in JSONL, and the positive usage receipt described above;
semantic Anthropic responses are accepted only when their provider-reported
concrete model equals the request.
Any change to those inputs invalidates that preregistration; prepare a new,
separately identified campaign before spending proposer budget.
The local write-once file is a reproducibility manifest, not an external
timestamp. For a confirmatory claim, publish or commit its digest before the
first proposer call so a post-outcome manifest cannot be presented as prior.

Unit tests may use static fixture proposals. Experiments should use the
cofibered LLM proposer and should report `NO_PROPOSALS` or `MISSING_LEG` rather
than silently falling back to unrestricted predicate search. Unrestricted
results remain valid, but they must be labeled as `UNRESTRICTED`.

## Minimal Smoke

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r bongard/requirements.txt
python3 -m py_compile bongard/crack_lab/*.py
.venv/bin/python -m pytest bongard
```

The unrestricted runner reuses normal Codex CLI authentication (`codex login
status` must succeed). The semantic Anthropic Messages runner separately
requires `ANTHROPIC_API_KEY.env.local` or `ANTHROPIC_API_KEY`:

```bash
python3 -u bongard/crack_lab/run_semantic_cone.py \
  --dataset-dir downloads/Bongard-LOGO \
  --source both --limit 5 --proposer anthropic --model sonnet
```
