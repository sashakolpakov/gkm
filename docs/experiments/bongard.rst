Bongard Visual Concept Induction
================================

Status: A1 failed; A2 invalidated; A3 failed; atomic attempts 1--2 failed; attempt 3 PRE-LIVE
------------------------------------------------------------------------------------------------

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

Atomic attempt two ran exactly once from commit
``d0864525146a05795c030674fa0159feb43913c1`` and tag
``bongard-atomic-successor-pre-smoke-20260806``.  Its historical input was the
nine-ID universe, digest
``sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed``,
with predecessor
``sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a``.
It appended exposure successor
``sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d``.

The journal closed with exactly 13 intents and 13 validated results: twelve
neutral support descriptions and one text-only atom proposal.  The proposal
receipt and schema were valid.  All ten observer questions ended in the ``?``
explicitly required by the prompt, but the shared soft-cue parser rejected
U+003F.  The exact error was ``invalid positive_description: soft cue
positive_description contains a forbidden prose character U+003F``, phase
``atom-proposal``, reason digest
``34b41a10ae89287ed97c875c6833047ff5896a7081debd144f484833292fe42f``.

No support-scoring call, formula, selection archive, query call, prediction,
label materialization, label reveal, or score occurred.  The run, journal
terminal, and command terminal persisted, and cold replay passed.  This is an
implementation-contract failure, not evidence about vision, predicate quality,
negation, or Bongard performance.  The selected task is consumed.  The
sanitized record has file SHA-256
``242ebc5914020a683a6f34a0b50688bf3190f4c4cbd6d345d15ebb5e775eb6b3``.

**Atomic attempt three is PRE-LIVE / PENDING.**  It binds the exact attempt-two
record and active predecessor
``sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d``.
The active universe
contains exactly eight IDs, digest
``sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9``.
Python is frozen around complete-release authentication.  A staged native
launcher performs a fixed non-Bongard structured-text transport preflight; all
stores must be pristine; and an exclusive seed-independent claim is persisted
beside the canonical predecessor path before secrets or exposure.  The journal
separately prevents resume or retry.  No live attempt-three result is claimed.

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
BD 3,434, HD 3,615).  Complete-release authentication hashed official-test
bytes, but no official-test task or panel was selected, exposed to the proposer
or scorer, evaluated, or scored.

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
       -> 1..12 pairwise-distinct exact affirmative observer questions
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

A phrase such as ``Does the panel contain a bird-like object?`` is therefore an
exact operational observer question, not a theorem about the pixels.
Each question is at most 192 UTF-8 bytes, has no outer whitespace, matches
``[A-Za-z0-9]+(?:[ -][A-Za-z0-9]+)*\?``, contains exactly one final ASCII
question mark, and receives no normalization or repair.  The same closed Python
policy rejects negation, disjunction, laundering, bundling, support-relative
language, and control/code text.  This surface guard is not proof of semantic
atomicity.  ``operational_nonmatch``
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
for interval evidence.  Python is the sole authoritative semantics.  It defines
predicates, the IR, evidence dispositions and projections, calibration,
synthesis, selection, evaluation, persistence, cold replay, admission and
benchmark decisions, and every scientific result or artifact ID.

Lean is neither imported nor required by the authoritative path.  A Lean or
other checker may consume only an already-frozen Python artifact and emit a
detached, non-authoritative sidecar; it may not propose a different predicate,
change an evaluation, admit a run, or alter an artifact ID.  Installing,
changing, failing, disagreeing, or deleting the sidecar must leave every
predicate, evidence value, formula, result, decision, replay, and ID unchanged.
Even a successful proof is conditional on empirical observations; it does not
verify the pixels-to-description step.

Information and exposure boundaries
-----------------------------------

The official release descriptor binds the archive, split file, 12,000-task
inventory, and extracted corpus manifest.  An exposure ledger records exact
task IDs and conservative semantic collision keys.  Basic numbered morphology
siblings share a key; all 20 instances of an Abstract ordered attribute pair
share a key.  A task is not fresh merely because its exact ID is new.

That integrity boundary hashes all release bytes, including official-test
bytes.  The model-use boundary is separate: no official-test task or panel is
selected for an episode, exposed to a proposer or scorer, evaluated, or scored.

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
official complete-corpus visual-semantic score currently exists.  Atomic
attempt one adds an operational wrapper failure with irrecoverable 0--29 call
count.  Atomic attempt two adds a cold-replayable 13-call implementation
contract failure before support scoring.  Both selected tasks are consumed.
The distinct eight-ID atomic attempt-three successor is **PRE-LIVE / PENDING**
and has no outcome in this snapshot.

See :doc:`reproduction` for the exact Stage-A command and current artifact
addresses.
