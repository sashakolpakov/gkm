# Bongard visual predicates

This directory is the active ShapeBongard V2 work. The goal is a headless
Codex proposer that infers one affirmative visual rule from six positive and
six negative support panels, freezes that rule, and predicts held-out panels
without seeing their labels.

There is no successful benchmark result yet. The current work is fixing the
representation layer before spending the remaining evaluation holdout.

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
24 more task IDs, four per split/family cell. The current ledger therefore has
180 exposed IDs and 10,020 exact-image-unseen train/validation tasks. This is a
large engineering pool, not 10,020 independent concepts: most tasks reuse
generator semantics already represented in the exposure ledger.

Strict reusable DRILL capacity is zero. The strict DEV reserve had 16 `bd`
tasks before the pilot and has 15 afterward. The pilot protected exact DEV task
IDs but failed to protect other tasks sharing their semantic disclosure keys;
one selected engineering task therefore disqualified one DEV unit. This is a
selector bug, not benchmark evidence, and the lost unit is not restored or
rerolled. The prospective selector now excludes the complete disclosure-token
closure (family plus morphology for ``bd``; pair plus attributes for ``hd``).
A metadata-only regression check kept every baseline-viable DEV task viable;
it opened no new pixels. The hardened v4 headless runner and v4 campaign
orchestrator are now implemented and fixture-tested, including an explicit
five-task exact-unused TRAIN mode for semantics-reused representation
engineering. At this pre-run snapshot, no real five-task plan has been frozen
or published and no model campaign has been executed. The 15-task strict DEV
cohort remains stopped because intended-concept expressibility is 0/15 in the
current v3 relational language; it is not a pending runnable plan.

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
and v4 campaign are implemented and fixture-tested. They have not yet been
used in a real Codex campaign. The packet binds the exact PNG digest,
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
permission to call again. These are fixture-tested implementation facts, not
evidence that a real plan or model campaign has run.

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

The qualified next run is therefore explicitly representation engineering,
not DEV: five exact-unused TRAIN tasks whose generator semantics are already
historically exposed. The hardened v4 runner and v4 campaign implement and
fixture-test this exact fixed-allowlist mode. The candidate set is the three
`big_small_*_triangles` tasks, `two_unbalanced_triangles`, and
`two_mirror_unbala_triangles`. At this pre-run snapshot no real plan has been
frozen and no model call or campaign execution has occurred; live admission
still requires the frozen union and expressibility checks. The exact-unused
`unbala_trapezoid_right_triangle` task is in the sealed semantic partition and
must not be substituted.

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
pre-pilot baseline of 10,044; the coverage pilot then gave the current 10,020.
Do not mix those snapshots.

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
