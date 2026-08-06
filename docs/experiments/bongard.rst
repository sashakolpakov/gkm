Bongard Visual Concept Induction
================================

Status: A1 failed; A2 invalidated; A3 failed; atomic N=1 failed operationally
-----------------------------------------------------------------------------

The active target is the complete official ``ShapeBongard_V2`` corpus: 12,000
tasks and 168,000 PNG panels.  The normalized primary split contains 9,300
training, 900 validation, and 1,800 test tasks.  Exact release identity is
pinned by ``bongard/data/shape_bongard_v2_release_v1.json``.

The first live atomic N=1 attempt is an operational failure, not a Bongard
result.  It ran from commit
``62ea577f5d86d109577f4f5e49b8b4866eb76c92`` and tag
``bongard-atomic-pre-smoke-20260806``.  The command persisted cache, config, and
exact-task exposure, so the selected task is consumed and will not be
rerolled.  It persisted no prediction and no terminal.  The runner was entered
and returned a typed ``AtomicSmokeRun``.  Fallback terminal construction then
tried to JSON-clone its frozen ``MappingProxy`` precommit.  Normal terminal
construction contains the same deterministic defect, although the surviving
error does not identify the exception that first selected fallback.  The exact
error was ``failed run precommit is not canonical JSON``; its reason digest is
``2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d``.

The underlying run status, phase, output, and successful model-call count are
irrecoverable.  The count is unknown in the inclusive range 0--29.  Since no
prediction was persisted, labels could not be materialized or revealed.  No
score, calibration, semantic, benchmark, or official-test claim follows.  A
sanitized record is
``bongard/data/atomic_smoke_n1_operational_failure_v1.json``.  Atomic stores
must have mode ``0700``.  An earlier setup launch rejected a ``0755`` cache
store before exposure and consumed nothing.

Attempt two is frozen as a distinct successor, not a reroll.  Its active
predecessor is the first attempt's exposure successor,
``sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a``.
That ledger must be exactly one canonical append after historical A3 ledger
``sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4``,
and the command also binds the exact incident and prior config/reason lineage.
After excluding the consumed first task, the active universe is exactly nine
IDs with digest
``sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed``.

The pinned native launcher is staged and authenticated before any selection,
episode, or label secret and before exact-task exposure.  Attempt two requires
a fresh empty mode-``0700`` call-journal store.  The journal durably writes a
bound header, an exact intent before each slot in the fixed 29-call schedule,
each validated result before the next intent, and its terminal before runner
return.  An existing header, open intent, or partial prefix cannot be resumed
or retried.  No live attempt-two result is claimed.

The first visual-semantic calibration experiment, A1, is terminal:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - A1 status
     - failed before scoring
   * - Command receipt
     - ``sha256:9aa247d9...40cb``
   * - Terminal failure
     - ``sha256:a130d9e6...65b83``
   * - Proposer funnel
     - 48 successful calls; 37 accepted soft, 10 direct-only, 1 parser rejection
   * - Scorer funnel
     - 37 transport errors / 37 attempts; 0 scores
   * - Label/calibration state
     - labels withheld; no fitted calibration, semantic accuracy, or negation evidence

A1's consumed seed was ``f9ee0fc4...f107857`` and its durable successor is
``sha256:99597cf6...a0f5d7a78``.  They remain historical identities and cannot
be reused.

A2 removed unsupported scorer transport-schema keywords while retaining exact
cue and witness validation in Python.  The protocol identity therefore changed;
A2 was not an A1 retry.  Its consumed identities were:

.. list-table::
   :header-rows: 1

   * - Field
     - Frozen value
   * - Protocol
     - ``sha256:2d9261c7...81ca``
   * - Fresh no-reroll seed
     - ``eb031fe1...5a40b1``
   * - Predecessor / durable successor
     - ``sha256:99597cf6...a0f5d7a78`` / ``sha256:9b7cb7ee...5b1ce8``
   * - Terminal result
     - no terminal artifact; invalidated by live source mutation

A2's successor exposure ledger and private cache preimage were persisted before
official panel bytes or model calls were released.  A concurrent agent then
edited ``bongard/typed_visual_proposal.py`` after protocol/cohort freeze.  The
live grammar no longer matched the frozen protocol, so the process exited
without writing a Stage-A terminal artifact.  The incident file digest is
``sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4``.
Process output showed 48 proposer and 34 scorer launches only; outputs were
lost, labels were not revealed, and no calibration, accuracy, or semantic
inference is valid.  The selected semantic groups remain consumed and the same
cohort may not be rerun.

A3 then completed the headless proposer/scorer path and exited 2 as a canonical
scientific failure.  Its exact reason was ``calibration score bins are
underpopulated: 1``.

.. list-table::
   :header-rows: 1

   * - Field
     - A3 result
   * - Command receipt / failure
     - ``sha256:2a019333...5681`` / ``sha256:cc1b86d7...31eb``
   * - Proposer funnel
     - 22/22 transport successes; 15 accepted soft, 6 direct-only, 1 parser rejection
   * - Scorer funnel
     - 15/15 transport successes; scores 0:8, 0.5:1, 1:6
   * - Lower bin ``[0, 0.75)``
     - 9 clusters; 1 affirmative
   * - Upper bin ``[0.75, 1]``
     - 6 clusters; 5 affirmatives
   * - Fit state
     - failed: the fixed minimum was 8 clusters per bin

Intended-bin orientation was 13/15 versus 2/15 for its exact complement.  At
the naive ``score >= 0.5`` threshold, orientation was 12/15 versus 3/15.
Negation did not win.  With only 15 scoreable claims, two bins of eight were
mathematically impossible; A3 exposed a recruitment/bin-power failure.  It
consumed 22 tasks and leaves 10,047 exact-unused train/validation IDs (FF 2,998,
BD 3,434, HD 3,615).  SEALED/test remained untouched.

The single typed-parser rejection came from matching ``def`` at the start of
the ordinary cue word ``defines``.  The expression now requires a complete
forbidden-keyword match.  That post-A3 fix does not revise the recorded result.

A later audit found that A3's launcher receipt authenticated the installed
JavaScript wrapper and reported ``codex-cli 0.146.0``, but not the native
client dynamically spawned by that wrapper.  The wrapper digest was
``134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477``.
This does not supply evidence of executable drift, but it means exact native
client bytes were not authenticated.  New runs prospectively repair the gap:
the production boundary opens the native binary without following links,
hashes and copies the same descriptor into a private executable, verifies the
reported version, and rechecks the staged identity.  Its pinned digest is
``sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02``.

Stage B did not run and is unauthorized by A1, A2, and A3.  No current receipt
permits a strict-disjoint DEV experiment.  The completed capacity audit found
exactly 24 BD + 0 constituent-disjoint HD = 24 DRILL units immediately before
A3, so the old
48-task design is impossible.  The earlier 28-unit upper bound failed to seed
HD constituent exclusions from the complete A2 ledger.  DEV against every
live-ledger exposure has 16 BD + 0 HD units.  The default 24-task request fails
before pixels or model
calls, and a 16-task BD-only pilot cannot meet the frozen 24-cluster minimum.
Any future design remains descriptive and sets
``dependence_design_authorized`` to false.  Official SEALED/test
visual-semantic execution is hard-disabled in both the CLI and benchmark API.

Exact v3 replay against A3's successor ledger now certifies strict DRILL
capacity zero: zero eligible tasks, zero eligible generator groups, and
``0 BD + 0 HD``.  The certificate digest is
``sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1``.
DEV remains exactly ``16 BD + 0 HD`` under that ledger.

The pre-A3 24 and post-A3 zero are strict independence-policy capacities, not
corpus size.
The official train/validation split contains 10,200 tasks; after A3, 10,047
exact task IDs remain unused (FF 2,998, BD 3,434, HD 3,615).
Calibration and evaluation were therefore using one unnecessarily strict
frame.  A future calibration design should use exact-unused training tasks,
account explicitly for shared generators, score both held-out task panels
before label reveal, and keep a separate semantic holdout for evaluation.  HD
evaluation partitions must be built from constituent attributes, not merely
ordered pairs.

A3 fixed its two-bin minimum at eight before its seed, pixels, or model output.
At 90% simultaneous confidence the associated Hoeffding radius was 0.480161,
making eight the smallest bin that could possibly decide against the frozen
0.5 boundary.  The high bin reached only six, so no fit or Stage-B authority
exists.

New Stage-A receipts are source-bound v2 records.  The runner rechecks the
complete executable Bongard Python source boundary around exposure, transport, replay, and
terminal serialization; post-exposure drift now writes a durable failed
receipt with labels withheld.  Identity-preserving caches reduced the same
synthetic Stage-A path from 161.15 s to 11.50 s and compact Stage B from
218.88 s to 51.10 s without changing canonical bytes or digests.

What was missing
----------------

The poor earlier results exposed a representation and experimental-design
failure, not a missing theorem prover.

The earlier PURE baseline reduced each panel to a small neutral raster feature
vector.  Coordinate-wise interval boxes threw away correlations between
preprocessing scenarios, a single centroid per class could not represent
multimodal near misses, and one feature-group per proposal ruled out many
cross-group concepts.  Those observables cannot reliably express concepts
such as ``bird-like object``, ``oblique angles``, ownership, contact, enclosure,
or part correspondence.

The replacement direct catalog still has no oblique-angle band, complete
point-contact signature, persistent part ownership, or bird-like predicate.
Those remain perception work; only the bounded soft path can currently name
such concepts operationally.

The A3 synthesis hole was equally concrete.  Its proposer recorded rich
descriptions of all twelve panels, but those descriptions were audit-only.  It
made one irreversible guess from zero to three direct catalog atoms plus at
most one bundled soft claim.  Synthesis merely lowered that guess, and the
scorer collapsed one to four cues by their minimum into ``0``, ``0.5``, or
``1``.

The atomic successor now uses the frozen descriptions as the only atom-proposal
input, records a complete atom-by-panel matrix, and deterministically selects a
positive conjunction before query release.  Its remaining representation gap
is not candidate search but the lossiness of one-sentence descriptions and the
absence of richer typed object, part, angle, topology, and relation facts.

The earlier PURE support diagnostic recorded 10 reversed outcomes among 132
executable support-panel outcomes.  It did not execute an A1 complement or
measure negation.  More generally, a bad predicate's complement can look better
when the predicate tracks a support-set correlate in the wrong orientation;
that does not repair the concept.  The current synthesizer therefore admits
only positive registered atoms and positive conjunctions.  There is no polarity
flip, no ``Not`` rescue, and a failed fit is not converted into a negative
observation.

The other hole was statistical.  Abstract examples with the same constituent
attributes are related, so treating them as independent observations and
applying a pooled Hoeffding claim was unjustified.  The current run is labelled
descriptive rather than inferential.  Future authorization would require a
frozen exact-key population, a post-freeze auditable seed, family-stratified
sampling without replacement, preregistered bins and thresholds, and an
explicit dependence or repeated-execution model.

The actual pipeline
-------------------

The system is not ``panel -> prose -> Lean -> truth``.  The new atomic smoke
separates empirical observation from deterministic synthesis:

.. code-block:: text

   12 support PNGs
       -> 12 isolated neutral vision descriptions
       -> one text-only proposer over labeled descriptions
       -> 1..12 affirmative single-phrase observer predicates
       -> 12 isolated one-panel calls covering the full atom matrix
       -> deterministic positive conjunction of at most four atoms
       -> frozen formula

   only after the freeze:
       2 query PNGs -> 2 descriptions -> 2 selected-atom observations
       -> durable joint prediction -> label reveal -> score
       -> model-free replay of all 29 causal receipts
       -> durable call-journal terminal before runner return

The atom proposer receives only the frozen descriptions and support labels,
not pixels.  Each description is bound to the panel bytes, neutral-description
protocol, validated receipt, run commitment, phase, and call ordinal.  The
one-panel observer then evaluates every proposed phrase, and every observation
is joined to the exact scorer producer, output, receipt, run, and call ordinal.

A phrase such as ``bird-like object`` is therefore an exact operational
observer question, not a theorem about the pixels.  ``operational_nonmatch``
may act as false only in an archive explicitly scoped to that observer.  Its
general semantic projection is ``indeterminate``, never
``certified_absent``.  The archive fixes
``calibration_authorized = false``, ``semantic_truth_claim = false``, and
``benchmark_claim_authorized = false``.  Calibrated-semantic atomic selection
is hard-disabled until Python can cold-validate a typed calibration artifact
and its interval rule.

Every observation has one of four dispositions:

``present``
   The registered measurement or calibrated soft protocol supports the atom.

``certified_absent``
   A direct or independently calibrated protocol supports absence.  The
   uncalibrated atomic observer cannot produce this disposition.

``indeterminate``
   The evidence interval or calibration does not decide the atom.

``error``
   Extraction, transport, identity, or verification failed.  Errors remain
   errors and cannot enter the denominator as negative predictions.

Witness IDs, intervals, units, producer versions, and content digests make an
observation replayable.  They are provenance, not proof that a high-level
description matches the pixels.

That description/matrix/selection path is now implemented in authoritative
Python.  It retains no ``Not`` and no polarity flip.  The remaining perception
work is to replace lossy free prose with richer typed object, part, angle, and
relation observations.  Any semantic claim additionally needs a powered,
independently frozen calibration design; the operational smoke does not supply
one.

Why Python is authoritative
---------------------------

The serialized Boolean IR is closed and unit-aware, and comparisons are safe
for interval evidence.  Python alone is authoritative for its typechecking and
predicate execution, calibration, cold replay, benchmark decisions, and
scientific artifact IDs.  This is the benchmark contract.

Lean or any other proof checker is optional.  It may only consume and
independently check a translation of an already-frozen Python artifact as a
detached sidecar; it may not propose a different predicate, change an
evaluation, admit a run, or alter an artifact ID.  Deleting the sidecar must
leave every result, decision, and ID unchanged.  Even a successful proof would
be conditional on the empirical visual witnesses; it would not verify the
pixels-to-description step.

Information and exposure boundaries
-----------------------------------

The official release descriptor binds the archive, split file, 12,000-task
inventory, and extracted corpus manifest.  An exposure ledger records exact
task IDs and conservative semantic collision keys.  Basic numbered morphology
siblings share a key; all 20 instances of an Abstract ordered attribute pair
share a key.  A task is not fresh merely because its exact ID is new.

The proposer receives labeled support panels.  Panel evaluators receive
neutral, candidate-independent single-panel views, not source paths, task IDs,
split metadata, or labels.  The formula is frozen before query pixels are
created.  Query predictions are committed before labels are revealed.  Cold
replay checks the bytes, identities, formula, dispositions, digest chain, and
score without calling a model.

These controls establish what computation occurred.  They do not establish
that a foundation model had never encountered the public corpus during
pretraining, nor do they turn an exploratory DRILL or DEV measurement into a
SEALED result.

Result policy
-------------

A1's valid terminal result is failure, not a pending value.  A2 is an
invalidated source-mutation incident, not a pending score.  A3 is a canonical
underpopulated-bin failure after successful proposer/scorer transport, not
evidence for negation.  Stage B is unauthorized by all three experiments.  No
official complete-corpus visual-semantic score currently exists.  The first
live atomic N=1 adds only an operational wrapper failure: exposure persisted,
no prediction or terminal persisted, the task is consumed without reroll, and
the successful-call count is irrecoverably unknown in 0--29.  The distinct
nine-ID, incident-bound attempt two remains pre-live and has no outcome.

See :doc:`reproduction` for the exact Stage-A command and current artifact
addresses.
