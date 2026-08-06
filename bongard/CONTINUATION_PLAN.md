# Bongard continuation plan

This is the current execution roadmap for the canonical visual track. It
replaces the old Phase A–D plan, which described the historical `crack_lab`
pilots. The objective is a reproducible complete-corpus benchmark with a
headless Codex proposer and a reusable leg library—not another score on
synthetic rerenders.

## Current foundation

Implemented and covered by focused tests:

- complete `ShapeBongard_V2` discovery, structural validation, official split
  normalization, exact release descriptor, and content-addressed task/panel
  manifests;
- two full decoding passes over all 168,000 official panels: every file is a
  single-frame 512×512 RGB PNG with no Pillow info keys and zero anomalies;
- a frozen historical-exposure reconstruction and official-corpus cohort
  projection: 3,868 tasks are semantically historically clean under the
  stated, limited evidence model, split by generator group into 2,769 drill,
  542 development, and 557 sealed tasks;
- a live filtering overlay at the checked-in thirteen-event campaign ledger
  head
  `sha256:65c8dd508f6c21e64b0c777a83159a470fbab12cfb8fee6adf588c0a9c400c8b`:
  1,777 of the frozen 2,769 drill tasks remain live-eligible and 992 have
  semantic collisions; thirteen of those also have exact-task collisions. The
  ledger records 19 tasks and 29 semantic keys, producing 190 effective
  exposed keys after policy blocking. The overlay digest is
  `sha256:5012df37f3cf273e0ce3d3696e574df7ec4ce458639cda79d340603a6d3e3320`,
  and the live-membership digest is
  `sha256:9db3805bded543e492103fa7f8198ca88c79bc8c71055459ac4a738356216589`;
  this overlay does not mutate the frozen cohort or certify unseen bytes;
- append-only exposure ledgers, historical-import support, deterministic
  drill/development/sealed partitions, and sealed-set checks;
- one four-disposition `Evidence[T]` type with provenance and uncertainty;
- soft semantic observations that cannot be mistaken for truth, plus a
  score-to-predictive-support bridge whose statistical bounds remain
  conditional on externally authenticated inputs, preregistration timing, and
  cross-cluster sampling assumptions;
- typed leg contracts with units, domains, codomains, transformation behavior,
  source digests, and a frozen registry;
- a closed positive IR containing only atoms, conjunctions, and justified
  disjunctions, with static digest-pinned calls and interval-safe comparison;
- an explicit backend boundary whose reference evaluator and cold replay are
  pure Python; formula, registry, and evidence JSON remain independent of the
  checker implementation;
- verifier-owned typed attachment and archive-preservation contracts;
- stripped candidate-source-to-normalized-implementation binding (or the
  selected bytecode fallback), a verifier-precommitted incumbent source, a
  mandatory direct formula call, and a candidate replay gate; this does not
  hash dependencies, globals, closure contents, or the runtime environment;
- support commitment, one fixed proposal followed by twelve isolated support
  re-observations, a replay gate, a final proposal freeze that binds the gate
  before query release, two-query prediction commitment before label reveal,
  tamper detection, and plain-JSON model-free replay;
- admission gates for nuisance preservation, Brier calibration versus a fixed
  baseline, near-miss contrasts, anti-memorization, full accepted-archive
  replay, and AST novelty that charges same-size rewrites;
- an episode protocol that exposes 6+6 support panels, fixes one proposal,
  requires exact alignment on twelve fresh support-replay calls, releases two
  opaque queries only after the final freeze, and treats indeterminate/error
  predictions as incorrect for headline scoring;
- a headless Codex transport that stages invocation authentication without
  recording it, stages an optional signed-envelope workspace policy cache whose
  exact bytes are bound as `sha256:...` or `absent`, and records transport
  failure as error rather than negative evidence; these receipts are content
  bindings, not provider/model attestations;
- a content-addressed episode-plan commitment and strict verifier that binds the exact
  release, split assignment, task manifest, all fourteen official panel byte
  preimages, support gate, vision receipts, verifier-owned canonical HYBRID
  recompilation, and cold replay. A non-empty exposure-ledger predecessor still
  needs its ledger or an external anchor for authenticity.

This foundation defines honest accounting. It does not itself provide a good
vision system or an official benchmark number.

## Scientific boundary

The canonical factorization is:

```text
panel -> frozen visual score/judgment -> typed empirical evidence
      -> closed predicate -> conditional verification
```

The first arrow is empirical. Prose such as “bird-like” or “oblique” specifies
a candidate claim but is neither an executable predicate nor evidence. A
score or judgment must retain its model, prompt/method, input digest, receipt,
and—when a score is calibrated—its predictive interval and calibration
record. The last arrows are mechanical. They establish what follows from
those recorded observations, not what is metaphysically true of the pixels.

Lean is not needed for the benchmark. The current reference typechecker,
evaluator, and cold replay are pure Python, while the serialized IR, registry
snapshot, and evidence records are checker-neutral. Lean may later cross-check
the closed predicate and attachment implication, but it must remain optional:
removing it must not change predicate identities, run artifacts, admission, or
replay. It is not a replacement for held-out perceptual validation.

This is an explicit portability goal, not merely an implementation detail:
Python predicates are the reference executable semantics. “Un-Leaning” a
future checker must be possible by removing that checker without translating
the predicate library or invalidating old artifacts.

## Execution sequence

### 1. Freeze the corpus and information boundary

1. Acquire the complete external `ShapeBongard_V2` archive.
2. Run `ShapeBongardCorpus.discover(..., require_complete=True)` and persist the
   corpus manifest digest.
3. Import every recoverable historical task/panel exposure from old runs,
   screenshots, galleries, prompt workspaces, and manual inspection. Unknown
   exposure time should remain explicitly historical rather than fabricated.
4. Select eligible IDs from the official training/validation pools only.
5. Use the frozen deterministic generator-group partition—Basic shape family
   or ordered Abstract attribute pair—and publish its digest before adaptive
   work. Never partition repeated task instances independently.

Gate: official counts and split regimes validate; all partitions are disjoint;
the ledger and partition are content-addressed; sealed access is rejected.

### 2. Establish honest baselines

Run fixed baselines before growing any new leg:

- constant/side-prior control;
- existing registered-leg composition only;
- soft-description baseline with frozen prompt and calibration;
- shuffled-side control;
- identity/filename leakage probes;
- near-miss pairs selected without sealed labels.

Report overall accuracy, determinate accuracy, abstention/error rates, Brier
score, and per-family/per-regime slices. Do not convert abstentions into the
more favorable class.

Gate: every result has a frozen corpus/exposure/protocol identity and cold
replay receipt.

### 3. Drill perception and grow one leg at a time

For each drill failure:

1. Diagnose whether the missing capability is perceptual, representational,
   compositional, or merely calibration.
2. Ask the headless proposer for the smallest positive explanation and typed
   missing leg. It may use prose observations, but cannot emit `Not`, choose a
   post-hoc polarity, or call unregistered code.
3. Attach the candidate at a verifier-issued typed boundary.
4. Test constructive positives, certified absences, deliberately ambiguous
   panels, runtime failures, required nuisances, and close counterexamples.
5. Compare calibration against the preregistered baseline.
6. Require candidate replay, bind its source to the exact registered
   executable, bind the incumbent source to the verifier precommit, and charge
   normalized-AST novelty.
7. Replay the complete accepted archive.
8. Promote atomically or leave the archive unchanged.

Prefer object- and relation-level legs that survive renderer changes over
dataset-name detectors. Candidate examples include contour/part structure,
angle bands, contact signatures, topology, symmetry, and calibrated soft
semantic prototypes. A prose phrase alone is not a promoted predicate.

Gate: every accepted leg improves a declared drill failure, passes all
admission gates, and preserves all accepted attachments.

### 4. Factor repeated glue

Track which leg compositions recur across tasks. When a repeated diagram has a
stable typed interface, propose a shared composite leg and compare:

- retained behavior with and without factoring;
- marginal AST charge;
- number of downstream uses;
- full archive replay;
- development performance under the same episode protocol.

This is the cumulative-abstraction test. “Cofibrant gluing” means a new cell is
attached along an explicit compatible boundary and the retained library is
preserved. It does not mean the proposer can declare arbitrary diagrams or
complexity credits.

Gate: the factored library is behaviorally equivalent on the archive, cheaper
on repeated uses under the declared accounting, and no worse on development.

### 5. Freeze model selection on development

Use development tasks for prompt choice, thresholds, calibration maps,
registry selection, and growth stopping. Freeze:

- proposer model and exact invocation;
- observer/model versions and prompts;
- accepted registry and source digests;
- typed attachment contract;
- episode seeds and scoring policy;
- query-release nonces/authority;
- archive-preservation contract;
- runtime/dependency manifest.

Gate: a fresh process reproduces every accepted development episode and the
selected registry without model calls during logical replay.

### 6. Open the sealed benchmark once

The sealed runner must:

1. verify the corpus, exposure ledger, partition, reusable admitted registry,
   and protocol digests before exposing pixels; the separate HYBRID task-local
   registry is constructed only after its support proposal;
2. make one proposer call on 6+6 support panels per episode and fix its
   positive claim, formula, and task-local registry;
3. make twelve fresh neutral single-image support observations and reject any
   reversal, mismatch, indeterminacy, or error;
4. bind the accepted support-gate digest into the final proposal freeze before
   either query panel is released;
5. expose two query observations with only neutral callback IDs and temporary
   `query.png` paths, never source IDs, source paths, or labels;
6. commit both predictions before revealing either label;
7. score present/certified-absent as positive/negative and score
   indeterminate/error as wrong;
8. emit the complete artifact chain and cold replay receipt;
9. make no adaptive code, prompt, threshold, or reusable-registry change
   afterward.

Report official family/regime slices and the number of tasks actually opened.
Do not extrapolate a subset score to 12,000 tasks.

Gate: every episode verifies, the supplied ledger head and its external anchor
establish that no sealed exposure predates the one-shot open, and an
independent cold process reproduces all committed predictions and scores. The
CLI's unseen check alone cannot establish that a supplied ledger head is the
latest authentic one.

## Why the old negation result is not a theory

Negation won because the old candidate language could orient a weak feature
after seeing labeled support, and because unresolved or failed perception
could become a false-like value. Flipping that value rewarded ignorance. The
new primary IR has no negation or polarity field, and only a constructive
certificate counts as absence. The question is now whether a positive,
frozen explanation survives fresh queries—not whether either orientation of a
feature can fit twelve seen panels.

## Current drill evidence

The old resolver-v1 live count of 2,609 was too permissive, not evidence of a
larger clean pool. It treated numbered Basic names as independent families and
missed morphology siblings such as `advanced_lamp3`/`advanced_lamp4`. Resolver
v2 removes a terminal number or `_newN` for a conservative second key and
blocks groups that cross cohort boundaries. Its policy digest is
`sha256:48598ae580a2f88aee7652d36fd386d54a8e4265b040bf1313f558508f47af9a`.
For HD, “unseen” means only that an ordered attribute combination is absent
from the bound history; its component attributes need not be new, and its 20
instances are one sibling group rather than 20 independent concepts.

Before v7–v9, the initial resolver-v2 train-and-drill overlay held 1,290
tasks across 161 semantic groups. At the current head the full drill overlay
is 1,777 live and 992 excluded, while the official train-and-drill scope is
1,268 live and 828 excluded across 158 resolver-v2 collision groups. The latter
overlay digest is
`sha256:338a2b3745169f2c9511043e43893b58f0cb6d8f320a50245fd5d17030f33185`
and its live-membership digest is
`sha256:c06e028ea669242ff423d3035d6c23ac653fc51b90f130e3413db0b5f60f1a2b`.
Those 158 groups are collision domains, not statistically independent samples.
The retrospective montage record and the v6, v7, v8, and v9 support releases
remove their related concept groups. The run-time `external_anchor` fields are null;
the later repository commit is only an after-the-fact publication anchor, not
preregistration or proof that the supplied ledger was the latest authentic
head.

The checked-in v5 current-protocol attempt is
`bd_advanced_lamp3_0000_v5.json`, file SHA-256
`e3fbe8f76290bb93f33def26c36b50f9ae451e43456a52e4796976a71662255a`.
It used one stable cache binding,
`sha256:6860e08631caee1357061bd727e93f7d200931b3bb2d925f873aea3d669d22f2`,
but ended `support_rejected` before query release and has no query
observations or run archive. Its immutable v3 observation/v1 gate records six
forward matches, one reverse match, and five parser errors.

At the raw-output layer, all six positives were `present`; five negatives
were `nonmatch`, and one negative was `present`. The five nonmatches exposed
a contract contradiction: the prompt/schema allowed or inferred a top-level
`reason`, while the parser required null. Observation schema v4 and support
policy v2 now make the reason optional and certificate-bound. The full
370-test suite covers the repaired implementation. Post-hoc re-evaluation of
the archived raw outputs yields 11 forward and one reverse match, so the result
would be `unsupported`, not accepted. The archived v5 result remains
`observer_failure`; it cannot be rewritten or salvaged.

The reverse match also matters scientifically. The prose rule “bent
double-ended arrow” is overbroad even after the parser correction. A
privileged, post-hoc action-program audit shows that the Basic target is one
precise nine-action template and the false-positive negative is a distinct
geometric near miss. That metadata is oracle-only: it was not available to the
proposer or gate and cannot supply a benchmark predicate. It identifies the
next missing leg instead—a frozen contour/template or prototype scorer that
operationalizes a prose candidate from pixels and separates that near miss.

The subsequent v6 attempt selected the first lexicographic task in the
then-current live-eligible list without prior pixel inspection. This is
inspection-unbiased selection, not random sampling. Its artifact,
`bd_advanced_lamp4-exist_quadrangle_five_lines12_0000_v6.json`, has file
SHA-256
`6a120eabd4efeeee60b5555cbb581d6cced3d33206bb0ed556e61a29fb213057`.
The support-release event
`sha256:b8fe3ea944d118058ac52e6f849ab5c1c1f6e08737f155e8b23f87569610877a`
advanced the campaign to an intermediate ten-event ledger head.

V6 is a benchmark-attempt failure. It ended `proposal_error` after
`plan_committed -> support_released -> proposal_failed`, before any accepted
proposal, support gate, query observation, or run archive. The failure category
was the blanket lexical rule `negative morphological complement`. The old
artifact schema did not retain the rejected raw payload or receipt, so the
decision itself is not independently auditable.

Two corrections are therefore required before another attempt: distinguish
constructive morphological language from logical negation, and persist the raw
payload and receipt for every rejected proposal. A future corrected run is a
new attempt. V6 cannot be retroactively repaired or promoted into a score.

V7, `bd_arc_cup_0000_v7.json`, has file SHA-256
`9801dbec0928f59667993a993b99f2cfcd6d5c02264bb10ef467ac98c427a462`.
Its event
`sha256:dbd578e1d3951837f25378721cf61e664eb96240e8f7c3fc108d1ff1db280a21`
produced ledger successor
`sha256:fc82fcebf4686c36f85f9efa0944ef4fc57b5da41dfccb19126c33b372c146dc`.
It ended `proposal_error` because DNS resolution failed before any Codex
response. It has no rejected proposal attempt, accepted proposal, support gate,
query observation, or run archive; this is a transport failure, not a score.

V8, `bd_asymm_bridge_0000_v8.json`, has file SHA-256
`ef50e35732c9a02d933ca1d7628589071270b06bc3d87fd0bb2543cdff16ccdb`
and status `complete`. It proposed “An enclosed region has a glyph-decorated
boundary,” operationalized as an enclosed cell with a boundary segment made of
repeated small circles, squares, triangles, or zigzag teeth. The support gate
aligned 12/12 and the two committed query predictions matched their revealed
labels. Every phase from `plan_committed` through `cold_replay_verified` is
present, and cold verification checks all 14 panel-byte preimages. The archive
digest is
`4f679fe175383a3ceb85333bf85f644dbe2a1ab69033747ae4b7d133893dc2ef`,
the chain digest is
`c2cefb76126cc18d5f5b4e39c4b506fc259cb6fdb02ebf1a7dfa666f92631f4d`,
and event
`sha256:25317bb78b0cf60b7585f59c93c7331c0f6743c3553ae044008b14b69d76fd35`
produced intermediate twelve-event ledger head
`sha256:7cf70dcb4e15aa8f0d8f82f4e5ff1e32f3018fb1f467061a5c947b0a5cf742d3`.

This is one successful integration episode, not an accuracy estimate. HYBRID
still uses an uncalibrated categorical self-observer: it asks the same model
family to propose a phrase and judge its matches. Mechanical replay establishes
what follows from the archived outcomes, not whether the outcomes are true of
the pixels. The missing pixels-to-score leg—a frozen, independently calibrated
visual measurement for open semantic claims and precise contour near
misses—remains the central theoretical hole.

V9, `hd_balanced_two-symmetric_transposed_0000_v9.json`, is a schema-v4
official-training HD ordered-combination attempt with file SHA-256
`6171b6bca42ffa6423d0e7e1ef753da325ef3d000e6f39d2ca28b5afccf8e655`.
Its phrase was “A matched opposing pair of lobes joined at one center,” with
the cues `paired_lobes`, `matched_geometry`, `central_junction`, and
`opposing_extents`. A stable transport binding covered all twelve calls, but
the support gate rejected it with nine forward and three reverse matches:
seven `present`, five `nonmatch`, no errors, and no indeterminate outcomes.
One positive missed `matched_geometry`; two negatives were false positives.
The phases stop at `proposal_frozen -> support_gate_rejected`, before any query
or run archive. Event
`sha256:63983c4c918b23d8a009bca43a3390a1cf876bf96894521760761552dd8c11f8`
produced the current thirteen-event ledger head above.

V9 is evidence that “matched” and “symmetric” need a quantitative
symmetry/shape-matching leg, not another vague categorical description. It also
exposes an artifact gap: a support-rejected schema-v4 run has no `run_archive`,
and its public outer plan stores only the support-commitment digest rather than
the nonce-bearing preimage. Current public `verify` therefore cannot fully
cold-bind and replay v9. The next outer schema must persist that preimage on
every post-proposal exit path; v9 must not be described as having v8's
fourteen-preimage verification scope.

## Historical results policy

Treat all existing symbolic, unrestricted, semantic-cone, grounded, soft, and
hybrid scores as archived pilots unless a result explicitly uses the complete
corpus manifest, exposure ledger, canonical episode protocol, and artifact
chain described above. In particular:

- the symbolic abstraction experiment assumes primitive observations;
- the action-program adapter bypasses vision and may use privileged metadata;
- synthetic rerenders test nuisance variation, not unseen concept instances;
- post-hoc predicates discovered after query inspection receive no benchmark
  credit;
- tiny local campaigns are engineering evidence, not population estimates.

The historical files remain physically present pending explicit deletion, but
they are excluded from the canonical reproduction path and preserved at the
annotated snapshot tag `pre-bongard-complete-rewrite-20260805`.

## Completion criteria

The migration is scientifically complete only when all of the following are
true:

- the complete official corpus and split validate by content;
- the complete official image audit validates the decoding boundary;
- all historical exposure is imported as far as recoverable;
- drill/development/sealed partitions are published and enforced;
- the headless proposer runs through the opaque episode boundary;
- vision observations retain provenance, uncertainty, and calibration;
- the accepted leg library grows only through typed atomic admission;
- every promotion replays the full accepted archive;
- both query predictions are committed before label reveal;
- the sealed run is one-shot, cold-replayable, and reported without inflating
  historical pilots into official results;
- documentation and manuscripts use the same claim boundary.
