Bongard Visual Predicates
=========================

Research question
-----------------

Can a headless visual proposer infer a reusable affirmative rule from six
positive and six negative support panels, commit to it, and classify held-out
panels without query leakage or post-hoc repair?

There is no successful benchmark result yet.  The pixel-to-observation and
closed-language layers, hardened v4 runner, and v4 campaign are implemented and
fixture-tested, but no real plan or model campaign has run at this pre-run
snapshot.  The reserved development holdout remains stopped at 0/15
intended-concept expressibility.

Corpus and exposure
-------------------

The pinned complete ``ShapeBongard_V2`` release contains 12,000 tasks and
168,000 PNG panels:

.. list-table::
   :header-rows: 1

   * - Family
     - Tasks
   * - Freeform (``ff``)
     - 3,600
   * - Basic (``bd``)
     - 4,000
   * - Abstract (``hd``)
     - 4,400

The primary split is 9,300 train, 900 validation, and 1,800 test.  Official
test is sealed from model use.  Complete-release authentication may hash those
bytes, but no test task or panel may enter proposal, threshold selection,
synthesis, or evaluation.

Immediately before the coverage pilot, 156 task IDs were exposed and
10,044 train/validation IDs were exact-image-unseen: 2,998 ``ff``, 3,431
``bd``, and 3,615 ``hd``.  They comprise 9,156 train and 888 validation tasks.
This is an engineering pool, not 10,044 independent concepts; generator
semantics are heavily reused.

The completed pilot exposed 24 additional tasks, four per split/family cell.
The current ledger has 180 exposed IDs and 10,020 exact-image-unseen
train/validation IDs.  Strict reusable DRILL capacity is zero.  Strict DEV had
16 ``bd`` tasks before the pilot and has 15 afterward.  The coverage selector
protected exact DEV task IDs but not shared semantic disclosure keys; the
selected ``bd_asymmetric_goldfish-unbala_three_intersect_circles2_0000``
therefore disqualified DEV witness ``bd_asymmetric_goldfish_0000``.  This is a
reserve-protection bug, not a result, and the unit is not rerolled.  The
prospective selector now excludes the complete disclosure-token closure
(``bd`` family plus morphology; ``hd`` pair plus attributes).  A metadata-only
regression preserved every baseline-viable DEV task and opened no new pixels.

The theoretical hole
--------------------

The tempting story is:

.. code-block:: text

   panel -> verbal description -> formal predicate -> proof

That story formalizes the wrong boundary.  Neither Python nor Lean can prove
from syntax that ``bird-like object`` is true of a PNG.  A checker can only
derive consequences of premises supplied by an empirical observer.

The useful decomposition is:

.. code-block:: text

   exact PNG bytes
     -> candidate-independent typed Python observations
     -> optional neutral calibrated vision tags
     -> closed positive same-binding Python predicate
     -> exact support gate
     -> durable freeze
     -> query release and joint prediction
     -> label reveal
     -> model-free replay

The first arrow is empirical and fallible.  It must expose uncertainty,
provenance, and scope.  The later arrows are deterministic.

Typed loop packets
------------------

Each support or query panel is observed independently.  The extractor receives
the exact PNG but not the task side, labels, candidate formula, or query role.
It binds the PNG digest and emits a complete registered packet rather than a
candidate-specific witness.

The packet retains every detected hole/loop and every unordered loop pair.  It
contains scenario-local object IDs, enclosed-area intervals, stable polygon
fits when available, edge-axis obliqueness, source ownership, contact evidence,
algorithm identities, and provenance.  Quantities are serialized as integers
with explicit units.

Typed scene fragments can be attached transactionally.  A gluing either adds
the complete, type-checked interface or adds nothing.  Reuse preserves object
identity and avoids recomputing observations; it does not waive verification.

Closed relational predicates
----------------------------

The current finite Python language has two explicit distinct roles and clauses
for polygon side count, directed enclosed-area ratio, minimum edge-axis
obliqueness, and optional point contact.  All clauses use the same ordered
object binding.

This same-binding rule matters.  Without it, a system can borrow a triangle
from one loop, an area ratio from another pair, and a contact from a third and
mistake the splice for one visual relation.

The synthesis language contains affirmative atoms and conjunctions only.  It
has no ``Not``, complement, disjunction, polarity bit, or reroll.  The proposer
gets one attempt.  Python evaluates the registered predicate on every support;
only all-positive ``present`` and all-negative ``certified_absent`` passes the
gate.

Evidence dispositions
---------------------

Every observation is one of four dispositions:

``present``
   A replayable affirmative witness exists.

``certified_absent``
   The registered observer has enough scoped evidence to rule the predicate
   out.

``indeterminate``
   The observer cannot decide within its certified scope.

``error``
   Extraction, transport, evaluation, or integrity checking failed.

A failed fit is not a negative.  This prevents negation from converting a
missing capability into apparent positive evidence.

Point contact
-------------

A complete positive point-contact observation requires two explicit loop
owners, one contact, four owner-labelled incident rays, both exterior gaps,
cyclic-order uncertainty, and provenance.  The current extractor can certify
some separations and represent positive signatures within its scope.

It cannot recognize the thick-stroke vertex attachment in atomic attempt
three.  Those contact observations remain ``indeterminate``.  Contact is
therefore optional and cannot be used as certified absence merely because a
fit failed.

Soft visual tags
----------------

Some concepts are not convenient pixel formulas.  The experimental v1
vision-tag layer admits finite neutral tags such as ``gestalt.bird_like`` and
``geometry.oblique_edges``.  The intended observer scores every neutrally
named loop before the candidate predicate exists.  The current envelope binds
the PNG, loop packet, complete object-by-tag inventory, integer score
intervals, prose, and caller-supplied prompt/model/protocol digests.  Those
opaque digests are not evidence that Codex executed.

The prose is audit material, not executable truth.  A closed tag predicate is
``present`` only when a calibrated lower bound clears a frozen threshold.
Version 1 is strictly presence-only: every low score remains
``indeterminate``, and v1 rejects absence authorization.  The envelope is
fixture-tested, but the neutral object-presentation transport, Codex receipt
attestation, calibration, and benchmark evidence do not yet exist.

Atomic attempt three, corrected
-------------------------------

The historical proposer asked ``Is a small triangle attached to a tilted
quadrilateral?``  Its live scorer marked all six positive supports present;
the negatives were three operational nonmatches, two present, and one
indeterminate.  Python stopped with ``NoExactSeparatorError`` before formula
freeze or query access.

A later deterministic forensic reported a perfect shape/ratio separator on the
twelve support panels.  That support-resubstitution result is reproducible.
The later 5+1-versus-5+1 and zero-support-separator report was a panel-mapping
error: held-out positive source index 4 and held-out negative source index 5
were substituted into support while two resolved supports were dropped.

On the exact archived support mapping, the base triangle/quadrilateral
area-ratio-at-most-1/8 predicate produces six positive ``present`` and six
negative ``certified_absent`` results.  Exhaustive support-only enumeration of
the historical 2,520-member contact-inclusive diagnostic relational
superlanguage finds four exact separators: area
threshold 1/12 or 1/8, each without obliqueness or with role-1 obliqueness at
least 5 degrees.  All omit contact.

Both actual held-outs are ``indeterminate`` under the 1/8 formulas.  Under the
1/12 formulas the negative held-out is ``certified_absent``, but the positive
held-out remains ``indeterminate``.  Exhaustive enumeration therefore finds no
candidate that separates all fourteen panels.  The support result is not a
score and provides no held-out generalization evidence.

The corrected v2 closed-language gate subsequently froze all 65,678
proposer-reachable positive predicates before reading any PNG and replayed the
twelve authenticated support panels as exact composite packets.  The gate
passes mechanically and finds exactly four support separators, all among the
1,260 contact-disabled relational members.  The 64,400 direct-count predicates
and 18 symmetry predicates add zero
A3 separators.  This changes neither the missing attachment observation nor
the generalization status: no held-out, query, or official-test pixels and no
model participated.

The stable result is
``bongard/data/a3_closed_language_gate_result_v2.json`` with record digest
``sha256:f9b6373df4dbe5d63807cf7e21be931db7ec0e9dfba106917df73d0e170a52d6``.

Coverage pilot result
---------------------

The 24-task coverage report has digest
``sha256:f78626c51b0af34cb0ccd96ed56041a51bcaeb453d3f26b10ea1ed1377542ae0``.
Extraction succeeded on all 336 panels.  Across three scenarios per panel it
recorded 17,876 loops, of which 10,354 were substantive.  Polygon and
obliqueness evidence was ``present`` for 4,516 loops and ``indeterminate`` for
13,360.  Among 267,197 unordered loop pairs, point contact was ``present`` 46
times, ``certified_absent`` 116,520 times, and ``indeterminate`` 150,631 times.
This measures observer coverage on semantics-reused engineering data; it is not
a proposer benchmark.

Closed-library ablation result
------------------------------

The historical 2,520-member contact-inclusive diagnostic v3 superlanguage was
replayed on every one of the 24 already-exposed pilot tasks.  It is broader
than the current 1,260-member proposer-reachable relational branch.  It found
**0/24** exact separators over all
seven panels per side.  Across all 168 deterministic paired leave-one-index
folds it found **0/168** exact 6+6 fits, so there were zero held-out
generalizers.  The best forward-oriented predicate on any task got only
**8/14** panels correct.

All 336 PNGs reproduced their authenticated loop-packet digests, extraction
reported zero errors, and 1,344 sampled evaluations matched the canonical
Python evaluator.  No proposer or model participated.  The result therefore
locates the failure in the current two-closed-polygon shape/ratio language and
its role domain, not in model search and not in the lack of a polarity flip.

The compact checked-in record is
``bongard/data/relational_library_ablation_24task_outcome_v1.json`` with record
digest
``sha256:ea6ee897513c22f1db8e656570e6572f2955855bbadb5caa39d8dc5dc8d423cd``.
It binds full-report output digest
``sha256:0a4b601ffc794a640175d2afda4f4b0d7f57fc980700bafbf09848ea4768c59b``.
This is resubstitution/library-coverage evidence, not a benchmark or a
generalization estimate.

Language audit
--------------

The rejected 15-task strict DEV cohort is also outside the current v3
language.  That language requires two substantive closed loops, polygon side
counts in the frozen 3--8 grid, and a directed area-ratio clause.  Its intended
concept expressibility on the cohort is 0/15; at most two tasks have partial
polygon overlap.  The other concepts require open arcs or bands, lamps, axes,
balance, transposition, or symmetry.  Running v3 there would measure accidental
correlates rather than concept induction.

This audit exposed an integration omission.  Deterministic component/hole,
skeleton endpoint/branch/cycle/crossing, curvature, and bilateral-symmetry
witnesses already existed, but the earlier relational-only runner did not
admit them.  The implemented repair is a finite closed Python tagged union over
those existing direct legs, the relational leg, and symmetry thresholds, plus
an exhaustive support-only oracle.  The oracle reports language failure
separately from model failure.
The first union is still panel-global outside its relational branch and cannot
conjoin branches, so object-bound cross-leg gluing remains unfinished.

Historical record
-----------------

The earlier runs remain useful diagnostics:

* A1 failed scorer transport and produced no calibration or accuracy.
* A2 was invalidated by concurrent executable-source mutation.
* A3 completed scorer transport but failed a preregistered minimum-bin count;
  Stage B did not run.
* Atomic attempt one failed terminal serialization after exposure.
* Atomic attempt two exposed a prompt/parser mismatch before scoring.
* Atomic attempt three's live soft predicate did not separate support, so the
  run stopped before query access.

The historical post-Stage-A-A3 exact-unused count was 10,047.  Three subsequent
atomic records produced the pre-pilot baseline 10,044, and the coverage pilot
produced the current 10,020.  None of these runs is an official benchmark or
evidence for a polarity flip.

Authority and current work
--------------------------

Python alone defines the canonical observations, predicates, decisions,
persistence, and replay.  Lean is not imported or required.  An optional
detached checker may inspect persisted canonical JSON, but its presence,
version, output, or failure cannot change artifact identity or a decision.

The loop packet, relational evaluator, contact uncertainty, and soft-tag schema
are fixture-tested.  Production neutral-tag transport remains separate future
work.  The hardened v4 headless runner and v4 campaign are implemented and
fixture-tested, including an explicit fixed five-task exact-unused TRAIN
semantics-reused engineering mode.  At this pre-run snapshot no real plan has
been frozen or published and no model campaign has executed.  The remaining
15-task non-test DEV run is stopped at 0/15 intended-concept expressibility;
there is no benchmark result placeholder.

The first metadata-only 15-task plan is rejected and was never executed.  Its
public support-index digest had only 49 possible preimages and disclosed both
held-out indices for every task.  A replacement would need a fresh private
schedule, a hiding commitment, official-manifest authentication for every
opened panel, source identities stored immutably in the plan, and an exclusive
durable claim journal that makes a crashed proposer call terminal rather than
retryable.  The v4 runner and v4 campaign now implement and fixture-test those
controls, but the 0/15 language audit means the rejected strict DEV cohort must
not be regenerated.

The implemented qualified mode is a fixed five-task exact-unused TRAIN run on
historically exposed generator semantics.  It is representation engineering,
not independent generalization.  Its closed-union and expressibility gates,
fixed allowlist, and sealed-task exclusion are fixture-tested; no real plan or
model campaign has yet run.  The sealed semantic partition remains untouched.
