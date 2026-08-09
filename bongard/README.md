# Bongard visual predicates

This directory is the active ShapeBongard V2 work. The goal is a headless
Codex proposer that infers one affirmative visual rule from six positive and
six negative support panels, freezes that rule, and predicts held-out panels
without seeing their labels.

There is now one completed five-task representation-engineering result. It is
not an official DEV/test benchmark or an unbiased generalization estimate.
The representation layer still has to improve before spending the remaining
evaluation holdout.

## Status

The pinned complete release contains 12,000 tasks and 168,000 PNG panels:

| family | tasks |
| --- | ---: |
| Freeform (`ff`) | 3,600 |
| Basic (`bd`) | 4,000 |
| Abstract (`hd`) | 4,400 |

The primary split is 9,300 train, 900 validation, and 1,800 test tasks. The
official test split is sealed: its bytes may be hashed to authenticate the
release, but no test panel is available to the proposer, synthesis, threshold
work, or evaluation.

Immediately before the coverage pilot, 156 task IDs had been exposed and
10,044 train/validation tasks remained image-unseen: 2,998 `ff`, 3,431 `bd`,
and 3,615 `hd` (9,156 train and 888 validation). The completed pilot consumed
24 more task IDs, four per split/family cell, leaving 180 exposed and 10,020
exact-image-unseen train/validation tasks. The completed five-task engineering
campaign then moved those counts to 185 and 10,015. This is a large
engineering pool, not 10,015 independent concepts: most tasks reuse
generator semantics already represented in the exposure ledger.

Those numbers are historical checkpoints. After the 2026-08-09 panel-soft
drill, the exact exposure successor contains 314 unique task IDs: 290 TRAIN,
24 validation, and zero TEST. That leaves 9,010 exact-image-unused TRAIN
tasks, 876 exact-image-unused validation tasks, and all 1,800 TEST tasks
sealed. Exact-image-unused still does not mean semantically independent, so
new engineering cohorts must be stratified by generator/disclosure family.

Strict reusable DRILL capacity is zero. The strict DEV reserve had 16 `bd`
tasks before the pilot and has 15 afterward. The pilot protected exact DEV task
IDs but failed to protect other tasks sharing their semantic disclosure keys;
one selected engineering task therefore disqualified one DEV unit. This is a
selector bug, not benchmark evidence, and the lost unit is not restored or
rerolled. The prospective selector now excludes the complete disclosure-token
closure (family plus morphology for ``bd``; pair plus attributes for ``hd``).
A metadata-only regression check kept every baseline-viable DEV task viable;
it opened no new pixels. The hardened v4 headless runner and v4 campaign
orchestrator implement the fixed five-task exact-unused TRAIN mode for
semantics-reused representation engineering, and that campaign has now run
once. The 15-task strict DEV cohort remains stopped because intended-concept
expressibility is 0/15 in the current v3 relational language; it is not a
pending runnable plan.

The first metadata-only 15-task public plan is rejected and must never be
executed. It published an unkeyed digest of the six selected indices on each
side. Because there are only 49 possible omitted-index pairs, that digest
revealed both held-out indices for every task by brute force. No DEV pixels or
model calls were made under that plan. The v4/v3 implementation now supplies
hiding commitments and fresh private schedules, but the rejected 0/15 DEV
cohort must not be regenerated merely because those controls exist.

The compact, non-runnable rejection record is
[`data/relational_headless_full_current_dev_20260807.rejection.json`](data/relational_headless_full_current_dev_20260807.rejection.json).
The leaking plan and its private schedule/cache files were deleted.

## The actual pipeline

The system is not `panel -> prose -> Lean predicate -> truth`. A proof checker
cannot establish that prose such as “bird-like object” is true of a PNG. The
hard part is the empirical first arrow.

The target protocol is:

```text
exact PNG bytes
  -> candidate-independent Python loop packet
  -> optional candidate-independent neutral vision tags
  -> one closed positive same-binding Python predicate
  -> exact support gate
  -> durable formula freeze
  -> query release and joint prediction
  -> label reveal
  -> model-free, tamper-detecting replay
```

Python is canonical. It defines evidence, predicates, evaluation, selection,
persistence, and replay. Lean is not required. An optional checker may consume
persisted canonical JSON after the fact, but installing, removing, changing,
or failing that checker cannot change an artifact identity or benchmark
decision. Historical records with Lean-named literals remain historical; they
do not define the current authority boundary.

The loop packet, relational query layer, closed-union path, hardened v4 runner,
and v4 campaign are implemented and have now been used in the completed
five-task Codex engineering campaign. The packet binds the exact PNG digest,
extractor identities, every retained hole/loop, polygon fits, area intervals,
edge-axis obliqueness, ownership evidence, and every unordered loop-pair
contact observation. All serialized quantitative values use integers and
explicit units.

The v4 runner implements the four previously missing controls. Hiding
schedule commitments replace low-entropy public schedule digests; every
opened PNG is authenticated against the official-release manifest; source
identities are frozen plan values; and a durable, exclusively locked task
claim prevents a crash or restart from making a second proposer call. A
claimed task with no sealed outcome is a terminal failure on recovery, not
permission to call again. The completed campaign exercised these paths and its
cold durable replay made zero proposer or model calls.

The closed relational language currently has factorized clauses for:

- polygon side count;
- directed enclosed-area ratio;
- minimum edge-axis obliqueness; and
- point contact.

That v3 language is far narrower than the reserved DEV tasks. It requires two
distinct substantive closed loops, polygon side counts in the frozen 3--8
grid, and a directed area-ratio clause; obliqueness is optional and the first
protocol disables contact. A metadata-only concept audit of the rejected
15-task DEV cohort found **0/15 intended concepts expressible** in that
language. Open bands and arcs, lamps, axes, balance, transposition, and most
symmetry concepts are outside it. At most two tasks have partial polygon
overlap. Running that cohort with v3 would measure accidental correlates, not
Bongard induction, so it is not authorized.

This exposed a concrete integration failure: deterministic component/hole,
skeleton endpoint/branch/cycle/crossing, curvature, and bilateral-symmetry
witnesses already existed in the repository, but the earlier relational-only
runner threw them away. The implemented repair is a frozen closed Python union
over those existing legs plus the same-binding relational leg, with an
exhaustive support-only expressibility oracle. Its first version remains only
a tagged sum with panel-global direct/symmetry atoms; cross-family conjunction
and object-bound direct/symmetry relations remain explicit follow-up work.

Two explicit roles must be distinct, and every clause is evaluated over the
same object binding. This prevents a triangle witness from one object, a size
witness from another, and a contact witness from a third from being spliced
into a fake conjunction.

The library is deliberately positive-only. Synthesis may choose affirmative
atoms and conjunctions that cover all positive supports. It has no `Not`, no
polarity bit, no disjunction, no complement rescue, and no reroll. A proposed
predicate that does not pass the exact support gate is a failed proposal.

## Four dispositions

Every empirical observation and predicate result is one of:

- `present`: there is a replayable affirmative witness;
- `certified_absent`: the registered observer has enough scoped evidence to
  rule the predicate out;
- `indeterminate`: the observer cannot decide within its certified scope; or
- `error`: extraction, transport, integrity, or evaluation failed.

A failed fit, missing observation, or model error is never Boolean false.
`indeterminate` and `error` therefore cannot be laundered into a negative
example and then “fixed” by negation.

## Soft predicates

Words such as “bird-like” and “oblique” can be useful, but their rigor comes
from their transport and calibration contract, not from turning the words into
a theorem.

The experimental vision-tag layer has a finite catalog (`gestalt.bird_like`
and `geometry.oblique_edges` in v1). A neutral observer is intended to score
every neutrally named loop object before seeing a candidate formula. The
current envelope binds the PNG, loop packet, complete object-by-tag inventory,
integer score intervals, prose, and caller-supplied prompt/model/protocol
digests. Those opaque digests are content fields, not proof that Codex ran.

The evaluator now requires the actual frozen calibration record for every
result and rejects an omitted or mismatched record. The prototype still must
not enter a benchmark: calibration and receipt digests remain caller-declared,
not a causally attested Codex transport.

The prose is audit material only; it is never executed. A closed tag predicate
is `present` only when its lower interval bound clears a frozen threshold.
Version 1 cannot certify negative visual semantics: every low score remains
`indeterminate`, and v1 rejects any purported absence authorization. The typed
envelope is fixture-tested, but the neutral Codex transport, object overlays,
receipt attestation, calibration, and benchmark run do not yet exist.

For deterministic notions, deterministic geometry remains authoritative. A
vision tag saying “oblique” can corroborate a geometric observation; it does
not override its uncertainty.

### First attested prototype-pair run

The preregistered recovery campaign completed as ``calibration_gap``. This is
a valid fail-closed engineering result, not a query-accuracy score: one rubric
description and 28 calibration observations made 29 receipted model calls,
then Python rejected three of the four required calibration bounds. No support
or query panel was released, no predicate was synthesized, and the sealed
campaign cold-replayed with zero model calls.

The result exposed two separate defects. First, the audit-prose firewall
rejected two otherwise valid observations solely because their descriptions
used the ordinary geometry word ``side`` or ``sides``; nondecisional prose
must not erase independently valid score cells. Second, even accepting those
cells leaves genuine visual collisions. Generator-identity absence is not the
same fact as absence of a broad prose appearance: other generators really can
look like “an open outline with a curved edge” or “a rounded outline with a
pointed appendage.” No scalar threshold separates the observed true and false
matches.

The next observer therefore needs object-local typed witnesses--contour
ownership, open/closed state, straight-span and smooth-arc count intervals,
branches, appendage topology, and marker evidence--with soft similarity as
calibrated accompanying evidence. Prose remains audit material and hypothesis
input. The executable predicate remains a closed Python value; Lean is
optional, removable, and cannot affect identity, selection, evaluation, or
replay.

The compact [sealed result](data/prototype_pair_targeted_engineering_20260807_live_v3_result_v1.json)
binds campaign record
``sha256:9851200d84132b36febe2dd1f029df37780fe56272ba8c636132f796fad45406``,
journal seal
``sha256:17ed15f2e902c1a7960f68cac67d4b885128a510da9308ca549c751564ba0ecf``,
and the exact source revision used for replay.

## The point-contact limit

The contact schema is intentionally demanding: two explicit loop owners, one
contact, four owner-labelled incident rays, both exterior gaps, cyclic-order
uncertainty, and provenance. The extractor can certify some separations and
can represent a full positive signature when it has one.

It cannot currently recognize the thick-stroke vertex attachment in atomic
attempt three. Those panels must remain `indeterminate` for contact. Contact is
therefore optional in synthesis and cannot be used as a negative merely
because the fit failed. Claiming otherwise would recreate the original bug.

## What atomic attempt three actually showed

The historical one-shot proposer asked:

> Is a small triangle attached to a tilted quadrilateral?

The live scorer produced six positive `present` results; on negatives it
produced three operational nonmatches, two `present`, and one `indeterminate`.
Python correctly stopped before formula freeze or query access because there
was no exact support separator.

A later deterministic forensic check found a shape/ratio separator on the
twelve actual support panels. A subsequent report saying that the base query
instead gave 5+1 versus 5+1 and that the library had no support separator was
itself wrong: it substituted the two held-out panels into the support set and
dropped two resolved supports.

The [canonical A3 forensic record](data/atomic_smoke_attempt3_relational_forensics_v1.json)
has record digest
`0487edf805fda6de40ecfc42add1d8bf95e435e0f6912f6e2fd8d2a25e89eb2a`.
It binds the archived support-call order to all fourteen source PNG hashes,
support/held-out indices, loop packets, base results, and all four formula
digests. It is the authority for the counts below.

With the archived source-index mapping restored, the base query--triangle as
role 0, quadrilateral as role 1, and directed area ratio at most 1/8--gives all
six support positives `present` and all six support negatives
`certified_absent`. Exhaustive enumeration finds four exact support
separators: area threshold 1/12 or 1/8, each either without an obliqueness
clause or with role 1 obliqueness at least 5 degrees. All four omit contact.

This is support resubstitution, not generalization. Held-out positive source
index 4 and held-out negative source index 5 are both `indeterminate` under the
1/8 base query, so that query's full seven-per-side result is 6 `present` plus
1 `indeterminate`, versus 6 `certified_absent` plus 1 `indeterminate`. Under
the two 1/12 support separators the negative held-out becomes
`certified_absent`, but the positive held-out remains `indeterminate`. None of
the historical 2,520-member contact-inclusive diagnostic relational
superlanguage separates all fourteen panels. Its contact-disabled,
proposer-reachable half contains 1,260 members and retains the same four
support separators. Point contact remains unresolved on the thick-stroke
attachment.

The corrected v2 closed-language gate now reproduces that result mechanically.
It froze all 65,678 proposer-reachable positive predicates before reading the
twelve authenticated supports, replayed exact composite packets, and found
exactly four forward support separators. All four are among the 1,260
contact-disabled relational members; the 64,400 direct-count members and 18
symmetry members add **zero** A3 separators. The gate therefore passes
its protocol check but does not improve the concept result or establish
generalization: it read no held-out, query, or test pixels and called no model.
The stable checked-in
[gate result](data/a3_closed_language_gate_result_v2.json) has record digest
`sha256:f9b6373df4dbe5d63807cf7e21be931db7ec0e9dfba106917df73d0e170a52d6`.

## Coverage pilot result

The exact-unused 24-task pilot completed on four tasks per split/family cell.
All 336 panels extracted successfully. Across three renderer scenarios per
panel it recorded 17,876 loops, of which 10,354 met the substantiveness floor.
Polygon and obliqueness observations were `present` for 4,516 loops and
`indeterminate` for 13,360. Among 267,197 unordered loop pairs, contact was
`present` 46 times, `certified_absent` 116,520 times, and `indeterminate`
150,631 times.

The content-addressed report is
`sha256:f78626c51b0af34cb0ccd96ed56041a51bcaeb453d3f26b10ea1ed1377542ae0`.
It is coverage evidence on exact-image-unseen but semantics-reused
train/validation tasks, not a proposer benchmark or a generalization score.

## Closed-library ablation result

The historical 2,520-member contact-inclusive diagnostic v3 superlanguage was
then replayed on all 24 already-exposed pilot tasks. It is broader than the
current 1,260-member proposer-reachable relational branch and remains valid as
a coverage ablation. The result is unambiguously bad: **0/24** tasks have an exact
separator over all seven panels per side, and **0/168** deterministic paired
leave-one-index folds have even an exact 6+6 fit. Consequently there are zero
held-out generalizers. The best forward-oriented predicate on any task gets
only **8/14** panels correct.

This is not a proposer failure: no proposer participated. It is not an
extractor crash: all 336 PNGs reproduced their authenticated loop-packet
digests, with zero extraction errors, and 1,344 sampled results matched the
canonical evaluator. It is a direct falsification of the current library's
coverage on this engineering cohort. The mandatory two-closed-polygon role
domain and shape/ratio vocabulary usually fail to make the positive panels
`present`; negation or a polarity flip would only conceal that missing
representation.

The compact checked-in outcome is
[`data/relational_library_ablation_24task_outcome_v1.json`](data/relational_library_ablation_24task_outcome_v1.json),
with record digest
`sha256:ea6ee897513c22f1db8e656570e6572f2955855bbadb5caa39d8dc5dc8d423cd`.
It binds the full write-once report
`sha256:0a4b601ffc794a640175d2afda4f4b0d7f57fc980700bafbf09848ea4768c59b`,
the diagnostic 2,520-query inventory, the exposure successor, and the
no-test/no-model/
no-negation restrictions. This remains resubstitution/library-coverage, not a
benchmark or generalization estimate.

## Completed five-task engineering campaign

The fixed campaign has now run on five exact-image-unused **TRAIN** tasks whose
generator semantics were already historically exposed. Its headline is
**2/5 jointly correct**. On the fixed ten-query denominator it scored
**4/10**; the other six queries remain unreleased or incorrect and stay in the
denominator. The two released tasks scored **4/4**, but that conditional number
is diagnostic only and must not replace either headline denominator.

Three tasks ended `support_rejected` and two ended `complete`; there were zero
terminal failures and zero rerolls. The two complete tasks,
`bd_two_mirror_unbala_triangles_0000` and
`bd_two_unbalanced_triangles_0000`, were jointly correct. The three
`big_small_*_triangles` tasks were support-rejected. Cold durable replay
accounted for the same two completions and three rejections with zero proposer
or model calls.

The failures locate the bottleneck before proof checking. The equilateral and
right-triangle proposals chose brittle area-ratio constraints: positive
evidence was indeterminate and at least one negative made the atom present.
The obtuse-triangle proposal chose `cycle_count == 1`, but five positives were
scenario-discordant or indeterminate. The two passing tasks instead selected
the robust `reflection_mismatch >= 500000` witness, stable on all twelve
supports and then correct on all four released queries.

A later support-only counterfactual recomputed all 65,678 frozen predicates on
the already released support packets, with no new pixels, model call, or query
access. The separator-count vector in frozen task order was **[0, 0, 0, 1,
1]**. It found zero support separators for each rejected task and exactly one
for each completed task: `reflection_mismatch >= 500000`, the predicate Codex
selected both times, with predicate digest
`0be38dc8ac08a4aab10e0b6a9fce3f11b730b809e1f77476802a173c70b12de8`.
The matrix digests for equilateral, obtuse, right,
mirror-unbalanced, and unbalanced are, respectively,
`eec8ce14b2158436bd461ea6fcafc57a33a28e232e0065e7bb48505c2ce861c9`,
`e5c641ba447f7e1c133331b8e09841162cb2d2e60b7c103bd11c897c9a08d0b2`,
`78abde91274995d46e0d3d817a3e351a091c9b8d8aee6f49b9dc5d42176b28d3`,
`c9fcd58f564e6f6bbe25b13516b95b697762c487420ca82ab4f3ac58147bf342`,
and
`586383b884c403347c12223603079480f402a5adc8822a479a34ca8f16da161d`.
This is post-hoc diagnosis, not a campaign metric. It shows that all three
rejections are current observer/language holes, not proposer misses: on this
cohort, support outcome perfectly tracked frozen-language expressibility and
the proposer found the unique separator whenever one existed. The practical
target is more robust candidate-independent witnesses and broader positive
language coverage. It is not a Lean bottleneck. Python is the canonical
predicate and replay authority; Lean is optional and can be removed without
changing identities or decisions.

The theoretical hole is now concrete. The proposal schema forced exactly one
predicate and had no typed `no_expressible_predicate` outcome. For the three
big/small triangle tasks, the closed IR contains no internal-angle,
triangle-class, side-ratio, or qualitative-size observable. Codex therefore
had to choose a nearby area-ratio or cycle-count proxy even though no member of
the frozen language fit support. Negating such a proxy can appear to improve a
score by exploiting its wrong direction, but it does not recover the missing
concept; this is exactly why polarity rescue remains forbidden.

The next pipeline should be:

```text
vision -> candidate prose plus object bindings
       -> calibrated four-disposition observations
       -> deterministic support oracle and viable version space
       -> typed language/witness gap if that space is empty
       -> Codex robustness/semantic ranking if it is nonempty
       -> freeze one verified positive predicate
       -> query release and replay
```

Internal-angle/triangle-class, side-ratio, and qualitative-size predicates are
the first representation additions. Soft terms such as “bird-like” may remain
useful hypotheses, but executable soft tags must be calibrated, provenance-
bound observers with `present`, `certified_absent`, `indeterminate`, and
`error`; raw prose never becomes truth by being translated into Python or
Lean.

The exact artifact bindings are:

- checked-in [plan](data/closed_visual_exact_unused_train_engineering_20260807.plan.json):
  `fa4e59fec47bef5f43cb530f3718d69b528059e5f219a1520498f2247ac3e3d3`;
- campaign report:
  `760448ab7d7be19325884e90e27a5eced3d4a5b9c7d356b7b6d70a4175ebc0c4`;
- durable replay:
  `0211f7b7480d580fc47dffaa1577a73a266a866e0f680446cad9272a5f30dcee`;
- exposure successor:
  `sha256:0d16900ac51f89885d1fb24c486b9b813f82c7863e1aa220da770460902d6d70`.

The checked-in compact
[result record](data/closed_visual_exact_unused_train_engineering_5task_result_v1.json)
binds those four source artifacts and the post-hoc support-only matrices; its
record digest is
`sha256:eab6dc107a21b12493307ef1070fe62534f728299113254dddd937a4f2498b4e`.

This is representation engineering, not strict DEV, official test, or an
unbiased estimate of generalization. The campaign moved the ledger to 185
exposed task IDs: 161 TRAIN, 24 validation, and zero official test. That leaves
10,015 exact-unused non-test tasks (9,139 TRAIN and 876 validation), but
exact-unused does not mean semantically independent. The official test remains
sealed, the strict DEV rejection and 0/15 language ablation remain in force,
and the exact-unused `unbala_trapezoid_right_triangle` task remains in the
sealed semantic partition.

## Historical experiments

Earlier Stage-A experiments are immutable diagnostics, not benchmarks:

| run | outcome |
| --- | --- |
| A1 | proposer transport ran; scorer transport failed; no scores or calibration |
| A2 | invalidated by concurrent source mutation; no terminal scientific result |
| A3 | scorer transport completed, but a preregistered calibration bin was underpowered; Stage B did not run |
| atomic attempt 1 | wrapper/precommit failure after exposure; no recoverable prediction |
| atomic attempt 2 | prompt/parser contract mismatch after descriptions and proposal; no scoring or query |
| atomic attempt 3 | live soft predicate did not separate support; stopped before query |

The historical post-Stage-A-A3 ledger had 10,047 exact-unused
train/validation task IDs. Later atomic records consumed three more, giving the
pre-pilot baseline of 10,044; the coverage pilot gave 10,020, and the completed
five-task engineering campaign gave the current 10,015. Do not mix those
snapshots.

No run above authorizes an official score, a claim that negation helps, or
access to the official test split.

## Code map

- `loop_geometry.py`, `point_contact.py`, and `loop_scene_witnesses.py` build
  candidate-independent, provenance-bound loop observations.
- `relational_scene.py` provides typed entities, ordered facts, and
  all-or-nothing additive gluing.
- `relational_visual_query.py` defines and evaluates the closed positive
  same-binding Python predicates.
- `composite_visual_packet.py` and `closed_visual_predicates.py` glue the
  existing direct, relational, and symmetry legs into a finite Python tagged
  union and distinguish language failure from proposer failure.
- `vision_tags.py` defines the neutral soft-observer packet, calibration, and
  presence-only closed tag predicate.
- `relational_coverage_drill.py` selects and measures an exact-unused
  train/validation engineering pilot while preserving strict DEV.
- `relational_headless_runner.py` implements the hardened v4 one-shot runner;
  `relational_headless_campaign.py` implements the v4 crash-safe campaign and
  its explicit five-task exact-unused TRAIN engineering mode.
- `atomic_smoke_*`, `benchmark.py`, and `semantic_*` preserve the earlier
  frozen experiment line and replay contracts.

## Verification

From the repository root:

```bash
python3 -m pytest -q bongard/tests
sphinx-build -W -b html docs docs/_build/html
make -C bongard/manuscript
```

Do not launch a live run merely as a smoke test. Exposure is persisted before
pixels or model calls, and consumed tasks are not rerolled.
