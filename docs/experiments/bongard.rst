Bongard Visual Concept Induction
================================

Status: A1 failed; A2 was invalidated by live source mutation
----------------------------------------------------------------

The active target is the complete official ``ShapeBongard_V2`` corpus: 12,000
tasks and 168,000 PNG panels.  The normalized primary split contains 9,300
training, 900 validation, and 1,800 test tasks.  Exact release identity is
pinned by ``bongard/data/shape_bongard_v2_release_v1.json``.

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

Stage B did not run and is unauthorized by both A1 and A2.  No current receipt
permits a strict-disjoint DEV experiment.  The completed capacity audit found
exactly 24 BD + 0 constituent-disjoint HD = 24 remaining DRILL units, so the old
48-task design is impossible.  The earlier 28-unit upper bound failed to seed
HD constituent exclusions from the complete A2 ledger.  DEV against every
live-ledger exposure has 16 BD + 0 HD units.  The default 24-task request fails
before pixels or model
calls, and a 16-task BD-only pilot cannot meet the frozen 24-cluster minimum.
Any future design remains descriptive and sets
``dependence_design_authorized`` to false.  Official SEALED/test
visual-semantic execution is hard-disabled in both the CLI and benchmark API.

The 24-unit number is a strict independence-policy capacity, not corpus size.
The official train/validation split contains 10,200 tasks; 10,069 exact task
IDs remain absent from the complete A2 ledger (FF 2,998, BD 3,456, HD 3,615).
Calibration and evaluation were therefore using one unnecessarily strict
frame.  A future calibration design should use exact-unused training tasks,
account explicitly for shared generators, score both held-out task panels
before label reveal, and keep a separate semantic holdout for evaluation.  HD
evaluation partitions must be built from constituent attributes, not merely
ordered pairs.

A3 is prospectively weaker than A1/A2.  After the code is frozen, it uses one
fresh no-reroll seed and the deterministic seed-ranked capacity.  The fixed
two-bin calibration requires eight clusters per bin.  At 90% simultaneous
confidence the associated Hoeffding radius is 0.480161, making eight the
smallest bin that can possibly decide against the frozen 0.5 boundary.  This
repair is fixed before the A3 seed or model outputs and cannot authorize Stage
B or SEALED.

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

The system is not ``panel -> prose -> Lean``.  It is:

.. code-block:: text

   labeled support pixels
       -> one typed positive proposal
       -> candidate-independent witness bundles for every panel
       -> direct registered atoms and, optionally, one soft visual claim
       -> valid, separately fitted calibration and four-disposition Python evidence
       -> a closed positive conjunction
       -> exact 12/12 support gate
       -> frozen formula and registry
       -> query release, prediction commitment, label reveal
       -> model-free, tamper-detecting Python replay

The proposal can use prose to name a concept, but prose is neither the
predicate nor proof.  Direct predicates consume typed panel observables.  A
soft claim such as ``bird-like object`` is evaluated by a blind one-panel
ordinal scorer against frozen references.  Only if a valid development
calibration has been fitted is its score mapped to an operational,
family-calibrated disposition.  It does not mean that Python proved the
presence or absence of a bird.

Every observation has one of four dispositions:

``present``
   The registered measurement or calibrated soft protocol supports the atom.

``certified_absent``
   The same declared protocol supports its operational absence.  This is not
   produced merely because fitting or transport failed.

``indeterminate``
   The evidence interval or calibration does not decide the atom.

``error``
   Extraction, transport, identity, or verification failed.  Errors remain
   errors and cannot enter the denominator as negative predictions.

Witness IDs, intervals, units, producer versions, and content digests make an
observation replayable.  They are provenance, not proof that a high-level
description matches the pixels.

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
invalidated source-mutation incident, not a pending score.  It has no terminal
artifact and supports no semantic inference.  Stage B is unauthorized by both
experiments.  No official complete-corpus visual-semantic score currently
exists.

See :doc:`reproduction` for the exact Stage-A command and current artifact
addresses.
