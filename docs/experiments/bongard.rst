Bongard Visual Concept Induction
================================

Status
------

The canonical Bongard track now has an official-corpus loader, exposure
ledger, typed evidence and leg contracts, a closed predicate language,
freeze/query/reveal artifacts, model-free replay, and atomic promotion gates.
It does not yet report a complete official-corpus score.

Pre-rewrite symbolic, action-program, unrestricted, semantic-cone, grounded,
soft, and hybrid pilots are preserved at the annotated Git tag
``pre-bongard-complete-rewrite-20260805``. Stale working-tree copies remain
physically present pending explicit deletion, but they are excluded from the
canonical reproduction path and are not results on the protocol described
here.

The claim boundary
------------------

The intended factorization is:

.. code-block:: text

   panel pixels
       -> frozen visual score or categorical judgment plus receipt
       -> calibrated predictive interval or archived empirical outcome
       -> typed, provenance-bearing witness
       -> closed positive predicate
       -> conditional mechanical verification

The first arrow is fallible perception. A visual model may report a
``bird-like object`` or ``oblique angles``, but the report is stored with its
input digest, producer/version, method, support interval, and witness IDs. It
is an empirical observation rather than a Boolean fact.

The prose names a candidate claim; it is neither the executable predicate nor
evidence. The development soft-predicate bridge starts from a distinct,
externally admitted ``FROZEN_VISUAL_SCORE`` packet. Given a caller-supplied
expected plan digest, exact development identities, and an externally
justified cross-cluster sampling assumption, it computes predictive-support
bounds. It checks content identities and clustering constraints, but not the
pixels-to-score operation, scorer or annotation authenticity, publication
time, or independence between clusters. The headless HYBRID baseline instead
archives an uncalibrated categorical model judgment for each panel. Its
12-panel support replay gate is a same-support orientation and consistency
check, not calibration or evidence of generalization.

The remaining arrows are mechanical. They can establish that a frozen formula
follows from recorded observations. They cannot establish that the model saw
the pixels correctly. Perceptual claims therefore require calibration,
nuisance tests, close counterexamples, fresh tasks, and explicit uncertainty.

The reference typechecker, evaluator, and cold replay are pure Python. The
serialized IR, registry snapshot, and four-disposition evidence are the
backend-neutral contract. Lean is not required or currently used. A future
Lean backend may independently cross-check that same contract, but benchmark
execution, admission, and replay must remain valid when Lean is absent. Such a
proof would still be conditional on the empirical visual witnesses supplied
to it. This removability is a design goal: Python predicates are executable
reference semantics, so adding and later removing Lean must not translate the
leg library or invalidate old run artifacts.

Official corpus and exposure
----------------------------

The target ``ShapeBongard_V2`` corpus contains 12,000 tasks, with seven
positive and seven negative PNGs per task: exactly 168,000 panel PNGs.

.. list-table::
   :header-rows: 1

   * - Family
     - Count
     - Description
   * - ``ff``
     - 3,600
     - Free-form
   * - ``bd``
     - 4,000
     - Basic shape concepts
   * - ``hd``
     - 4,400
     - Abstract/compositional concepts

The normalized primary split contains 9,300 training, 900 validation, and
1,800 test tasks. The test regimes are ``FF`` (600), ``BA`` (480), ``CM``
(400), and ``NV`` (320).

Exact release verification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Structural completeness does not prove release identity. The checked-in
``bongard/data/shape_bongard_v2_release_v1.json`` descriptor pins the
following exact values:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Object
     - Bytes
     - SHA-256 content address
   * - ``ShapeBongard_V2.zip``
     - 1,762,748,636
     - ``sha256:8c5542ac7b9ce8a6a14d157a0656dbde9da5b7843424eade4bd653759d9a27d0``
   * - ``ShapeBongard_V2_split.json``
     - 442,720
     - ``sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230``
   * - Sorted 12,000-task inventory
     - --
     - ``sha256:4503ae6b40dc7b34520eb5b8a4cca6ff8153635df0f42db5f6715cc349602dd0``
   * - Extracted corpus manifest
     - --
     - ``sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``

The descriptor's canonical digest is
``sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b``.
It also pins upstream commit
``9df7c78ee9c6a2ff041b48d9ed407359aac259c3``.

Run the exact check against both the downloaded archive and extracted tree:

.. code-block:: bash

   .venv/bin/python -m bongard inventory \
       --corpus downloads/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --require-complete \
       --official-release \
       --archive downloads/ShapeBongard_V2.zip \
       --out results/bongard/official-inventory.json

A successful command proves that archive name, size, and digest; split name,
size, and digest; all family/split/regime counts; the sorted task inventory;
and every panel-derived corpus-manifest commitment match the descriptor. The
canonical result exposes ``task_count``, family and split counts,
``manifest_digest``, ``split_source_digest``, and an ``official_release``
object with the pinned release commitments.

``bongard.corpus.ShapeBongardCorpus`` accepts the released ``images/`` tree
and equivalent generator ``png/`` tree. It validates structure, panel counts,
PNG signatures, official counts, split disjointness, and content hashes. The
small generator/gallery checkout used by old experiments is not the complete
archive. Only the descriptor-backed command establishes exact official
identity.

Full image audit
^^^^^^^^^^^^^^^^

Exact byte identity and successful image decoding are separate claims.
``bongard.image_audit.audit_corpus_images`` checks every panel as a non-symlink
regular file, rejects malformed PNG framing and trailing bytes, asks Pillow to
verify the container and load every frame, and records mode, dimensions,
metadata/info keys, and frame count. It reads and hashes each source once into
a bounded-memory spool, decodes the frozen snapshot, and performs a final
filesystem pass to detect ordinary mutation during the audit.

The complete exploratory pass and an independent strict pass both completed on
the pinned release. The strict result is checked in at
``bongard/data/shape_bongard_v2_image_audit_v1.json``.

.. list-table::
   :header-rows: 1

   * - Property
     - Complete-corpus result
   * - Tasks / panels
     - 12,000 / 168,000
   * - Total compressed panel bytes
     - 1,948,958,314
   * - Container / mode / size
     - PNG / RGB / 512 by 512
   * - Frames / Pillow info keys
     - One / none, for every panel
   * - Anomalies
     - Zero

The canonical report digest is
``sha256:d3485ada3605d708db82fbcfe6ecfc73506ce51ed85fcd1ce6ccd798e3bff9f8``.
The decoded-property summary digest is
``sha256:6feea60173c92a1357ffafbeecd78171c3455b3950a31229517ea07c6f03e811``;
the separately accumulated source-content summary digest is
``sha256:31f03303673e31a1a05f84ddd50621963ce4c73c1ab11073118c4905893389c5``.

The exploratory pass deliberately supplies no guessed image properties:

.. code-block:: python

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

The canonical data-only report contains task, panel, and byte totals; family,
format, mode, size, info-key-set, and frame-count distributions; stable
content and property summary digests; the bound corpus-manifest digest; a
bounded anomaly sample; and its own digest. It contains no paths, decoded
pixels, or Pillow objects.

The independent strict pass supplies the observed values explicitly:

.. code-block:: python

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

These values are explicit experiment inputs rather than library defaults.
Non-strict mode records mismatches as anomalies. Strict mode raises
``ImageExpectationError`` and leaves the complete report on
``exception.report``.

Historically clean cohorts
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspect the frozen historical-exposure classification before sampling:

.. code-block:: bash

   .venv/bin/python -m bongard cohorts \
       --corpus downloads/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --require-complete \
       --split train \
       --cohort clean \
       --limit 50 \
       --out results/bongard/train-clean-cohorts.json

The report includes its qualification, source-seed and split-index digests,
scope, counts, membership digests, and a bounded task-ID sample. This is a
metadata-only historical claim. It does not certify that official panel bytes
were unseen by people or absent from foundation-model pretraining.

On the complete release, 3,868 tasks are semantically historically clean:
1,328 Basic tasks from unused shape families and 2,540 Abstract tasks from
unused admissible attribute pairs. All 3,600 Freeform tasks remain
``indeterminate`` rather than being called clean. Abstract tasks are
partitioned by ordered attribute pair, keeping all 20 sibling instances
together. The combined clean pool contains 2,769 drill, 542 development, and
557 sealed tasks. The checked-in summary is
``bongard/data/shape_bongard_v2_cohort_summary_v1.json`` and has digest
``sha256:55de04a582ffa3a4fbf26466ab88f265ddd7839ae10004210cca4d9ffa4f8e9d``.
For HD, the historical claim concerns an ordered attribute combination, not
novel primitive attributes: each component may already be familiar, and the
20 generated instances are one semantic sibling group rather than independent
unseen concepts.

The frozen 2,769-task drill cohort is not a live availability count. Against
the checked-in thirteen-event campaign ledger head
``sha256:65c8dd508f6c21e64b0c777a83159a470fbab12cfb8fee6adf588c0a9c400c8b``,
the resolver-v2 overlay leaves 1,777 drill tasks and excludes 992 through
semantic-key collisions; thirteen are also exact-task collisions. The ledger
records 19 tasks and 29 semantic keys, producing 190 effective exposed keys
after policy blocking. The overlay digest is
``sha256:5012df37f3cf273e0ce3d3696e574df7ec4ce458639cda79d340603a6d3e3320``
and the live-membership digest is
``sha256:9db3805bded543e492103fa7f8198ca88c79bc8c71055459ac4a738356216589``.

The older resolver-v1 count of 2,609 was an overclaim because numbered Basic
generator names were treated as distinct families. Resolver v2 adds
conservative morphology-cluster keys obtained by removing a terminal number
or ``_newN``. Thus ``advanced_lamp3``/``advanced_lamp4`` and
``bird2``/``bird7`` collide, and a cluster is blocked if a sibling was exposed
or if siblings cross frozen cohort boundaries. The resolver-policy digest is
``sha256:48598ae580a2f88aee7652d36fd386d54a8e4265b040bf1313f558508f47af9a``.
Before v7–v9, the initial resolver-v2 training intersection contained
1,290 tasks across 161 semantic groups. At the current head, the official
train-and-drill scope contains 2,096 tasks: 1,268 live and 828 excluded across
158 resolver-v2 collision groups. Its overlay digest is
``sha256:338a2b3745169f2c9511043e43893b58f0cb6d8f320a50245fd5d17030f33185``
and its live-membership digest is
``sha256:c06e028ea669242ff423d3035d6c23ac653fc51b90f130e3413db0b5f60f1a2b``.
Those 158 groups are collision domains, not statistically independent samples.
The montage record and v6, v7, v8, and v9 support releases remove their related
groups. The overlay neither changes the frozen cohort nor certifies unseen
panel bytes. Run-time ``external_anchor`` fields are null; the later Git commit
is an after-the-fact publication anchor, not preregistration or proof that the
ledger was the latest authentic head.

For an official CLI run, ``bongard.exposure.ExposureLedger`` and the semantic
cohort check are used together. With ``--require-unseen --cohort``, the runner
checks exact and semantic collisions and writes one task-level event before
support release; that event makes all panels of the task non-unseen. The
generator-level Basic-family and Abstract-pair partition lives in the frozen
historical/cohort data rather than the generic exact-ID partition. Direct API
reads, manual inspection, and reuse of a stale ledger head are not intercepted.
An upstream regime called ``novel`` is not fresh if local history says it was
already seen.

Abstract v2 keys remain exact ordered pairs. They certify only that the
ordered combination is absent from the bound history; they do not make the
component attributes new or make the 20 generated instances independent.

There is no universal image normalization. Each visual leg declares the typed
view it consumes: literal ink for style-bearing properties, carrier shape when
stroke and raster nuisances may be quotiented out, or a relational view for
objects, contacts, containment, repetition, symmetry, and ownership. Applying
one global cleanup would erase concepts that genuinely depend on line width,
texture, or rendering detail.

Four dispositions
-----------------

Every registered leg returns exactly one ``Evidence[T]`` disposition.

.. list-table::
   :header-rows: 1

   * - Disposition
     - Meaning
     - Negative evidence
   * - ``present``
     - A value or witness was produced
     - Only after its atom is evaluated
   * - ``certified_absent``
     - A declared procedure established non-existence
     - Yes
   * - ``indeterminate``
     - Available evidence cannot decide
     - No
   * - ``error``
     - Implementation or contract failure
     - No

This prevents a failed fit from becoming a negative classification. Soft
semantic observations cannot be coerced to ``bool``. A calibrated IR atom
must compare the entire recorded interval to its threshold; an interval that
straddles the threshold is indeterminate.

The generic HYBRID observer emits the wire-level outcome ``nonmatch``. Its
archived payload can certify only that the fixed model procedure returned a
nonmatch for the frozen operational claim; it does not mechanically certify
that the semantic property is absent from the pixels. Pixel-level absence
requires a dedicated registered certifier or a calibrated interval wholly
below its frozen threshold.

Why negation used to win
------------------------

The historical search could evaluate a weak feature on labeled support and
choose whichever polarity scored better. Some unresolved structures also
fell through to false-like values. Negation then turned perceptual ignorance
into apparent evidence and received the score of the complementary rule.

That result exposed two protocol defects, not a deep law of Bongard concepts:

* orientation was selected after seeing support labels;
* absence, uncertainty, and implementation failure were not kept distinct.

The primary IR now has no ``Not`` node and no polarity flag. Its closed syntax
is an atom, a conjunction, or an explicitly justified disjunction. The general
proposal schema can represent positive conjunctions, but the canonical
headless HYBRID CLI currently compiles exactly one positive ``hybrid_claim``
atom. ``AnyOf`` is retained for verifier-owned library formulas with an
explicit semantic justification. Calls are static references to exact registered leg
versions and contract digests. Every scalar leg also declares an
``AffirmativeRelation``: the
specific subset of ``at_least``, ``at_most``, and ``between`` that expresses
more of its positive claim. The IR rejects an undeclared inequality direction.
Thus a low closure-error leg may legitimately declare ``at_most``, but a
synthesizer cannot try both signs of an unrelated score and retain the winner.
Non-scalar witness legs admit only ``present``. Units and interval comparisons
are checked mechanically. A weak predicate can still fail, but it cannot be
rescued by redefining which side is positive.

Typed legs and cumulative growth
--------------------------------

A leg is a reusable observation procedure with a declared domain, codomain,
unit, implementation digest, cost, and behavior under named transformations.
The verifier freezes the registry before issuing an attachment contract.
For the reusable admitted-registry track, the protocol additionally requires
that freeze before support. The generic runner does not enforce that timing;
the canonical HYBRID path creates one task-local empirical leg after support
and freezes it before support replay and query release.

Growth follows a PowerPlay-like loop:

#. Select a drill failure that the retained registry cannot solve honestly.
#. Propose the smallest typed leg or composition that attaches at the existing
   boundary.
#. Test constructive cases, declared nuisances, calibration against a fixed
   baseline, near-miss contrasts, and anti-memorization challenges.
#. Charge novelty from normalized AST structure, including same-size rewrites.
#. Replay every previously accepted attachment.
#. Promote one immutable archive extension only if every gate passes.

The cofibrant-gluing analogy refers to this explicit attachment boundary and
archive-preserving extension. It does not give a proposer authority to invent
its own complexity, omit a load-bearing interface, or treat a diagram as
evidence. Repeated stable compositions should be factored into shared legs so
marginal novelty falls with reuse.

Freeze, query, reveal, replay
-----------------------------

One canonical episode uses six positive and six negative support panels. One
proposer call fixes the claim, formula, and task-local registry. Twelve fresh,
isolated, neutral single-image observations replay that fixed claim on support;
all twelve must align. The final proposal freeze binds the accepted gate before
either query panel is released. A successful episode therefore makes fifteen
model calls: one proposal, twelve support replays, and two query observations.
Query objects expose only neutral callback IDs and a temporary ``query.png``
path: no source task ID, filename, filesystem path, or label crosses the
proposer/observer boundary.

.. code-block:: text

   support byte commitment
       -> fixed positive claim + task-local formula/registry
       -> 12 isolated support observations + replay-gate artifact
       -> final formula/attachment/registry freeze binding the gate
       -> two distinct unlabeled query blobs
       -> committed atom evidence and both predictions
       -> reveal both labels
       -> cold replay receipt

Every artifact contains or is bound by the canonical SHA-256 digest of its
parent. Both predictions are committed before either label is revealed.
Plain-JSON artifact replay reconstructs the closed formula and cached atom
dispositions without a vision model, proposer, or leg implementation. Full CLI
verification additionally reparses the proposal and canonically recompiles the
verifier-owned HYBRID formula, registry snapshot, source and operational
identities, and attachment contract without invoking the model. Any changed
gate evidence, parent digest, query byte, atom result, prediction, or label
fails verification.

That digest chain is self-consistent, not self-authenticating. Cold
verification therefore requires a raw run-file SHA-256 anchored outside the
file itself:

.. code-block:: bash

   .venv/bin/python -m bongard verify \
       --run results/bongard/episode.json \
       --corpus downloads/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --archive downloads/ShapeBongard_V2.zip \
       --expected-sha256 "$EXPECTED_RUN_SHA256"

``EXPECTED_RUN_SHA256`` must be exactly 64 lowercase hexadecimal characters
copied from an independent write-once record, signed manifest, or committed
publication. Deriving it from the run file at verification time does not
provide an external anchor. Canonical verification also checks the trusted
release descriptor, official split assignment and task manifest, an exact
bijection to all fourteen official panel byte preimages, canonical HYBRID
compilation, the support gate, every vision receipt's internal hashes and
request/response bindings, and cold logical replay. Codex receipts are not
provider signatures and may record that JSONL omitted the reported model. The
cold verifier also cannot authenticate the unavailable history behind a
non-empty exposure-ledger predecessor. Proposal-failure records are not
replayable.

Admission and reporting
-----------------------

Promotion requires all of the following:

* typed attachment to the frozen registered-leg boundary;
* binding of stripped submitted candidate source to normalized implementation
  source (or its selected bytecode fallback), a required direct formula call,
  and binding of incumbent source to a verifier precommit; this does not hash
  dependencies, globals, closure contents, or the runtime environment;
* required nuisance coverage and pass rate;
* calibration improvement over a fixed baseline;
* near-miss contrast performance;
* anti-memorization performance with zero train/query overlap and no forbidden
  identifier hit;
* replay of the candidate on the frozen candidate suite;
* exact replay of the complete accepted archive;
* source that parses and stays within its AST novelty allowance.

The decision is atomic. A rejection contains no next archive, and an accepted
decision applies only if the archive digest is unchanged since admission.
The receipt objects are checked data contracts, not signatures; production
evaluation must issue them inside the verifier-owned process or authenticate
them externally.

Report perceptual calibration and dispositions, episode accuracy, and library
growth separately. Indeterminate and error predictions remain visible but
score as wrong in headline query accuracy. A subset result must state the
number and identity of opened tasks; it must not be described as a 12,000-task
benchmark.

One positive and one negative panel are withheld in every current episode. If
the caller-supplied seed makes their order uniform and independent of the
predictor, a no-vision policy that assigns opposite labels to the two opaque
slots has 50% expected puzzle and image accuracy, not the 25% baseline for two
independent binary labels. The code deterministically derives order from the
seed but does not enforce randomness, secrecy, or external preregistration;
cold verification has only the seed digest. Reports must state this conditional
paired baseline and include per-image and per-side accuracy over multiple
independently selected semantic groups.

A pre-current-protocol v1 drill on the training task
``bd_trapez_parallelogram_0000`` classified both held-out query panels
correctly (2/2). It used the old corpus identity and predates exact official
release binding, the current receipt/schema checks, canonical HYBRID
recompilation, and the 12-panel support gate. The checked-in complete-release
v2 completed drill scored 1/2.

The first checked-in current-protocol v3 drill attempt,
``hd_exist_quadrangle-exist_sector_0000_v4.json``, has file SHA-256
``bf60a36bc7a48e61c61c8de2153753fa2996db54eacf53fe4c861bf06a9b4f41``.
It ended ``support_rejected`` before query release, with no query
observations or run archive and zero determinate query outcomes. All twelve
support observer calls otherwise succeeded, but the verifier converted every
raw judgment to ``TransportIdentityError`` because the supposedly stable
``cloud_config_bundle_cache_binding`` changed across calls. Before that
conversion, the raw judgments fit 10/12 support labels: all six positives were
``present``, while four negatives were ``nonmatch`` and two were
``present``. This exposes a transport-identity protocol bug; it is not a
held-out score.

The checked-in v5 drill attempt,
``bd_advanced_lamp3_0000_v5.json``, has file SHA-256
``e3fbe8f76290bb93f33def26c36b50f9ae451e43456a52e4796976a71662255a``.
Its proposal and support calls share one cache binding,
``sha256:6860e08631caee1357061bd727e93f7d200931b3bb2d925f873aea3d669d22f2``,
so the v4 transport-identity defect is absent. It nevertheless ended
``support_rejected`` before query release, with no query observations or run
archive and zero determinate query outcomes. The immutable artifact's formal
v3 observation/v1 gate records six forward matches, one reverse match, and
five parser errors.

The raw outputs were six positive ``present`` judgments, five negative
``nonmatch`` judgments, and one negative ``present`` judgment. The five
nonmatches exposed a prompt/schema/parser contradiction: a top-level
``reason`` was allowed or inferred, while the archived parser required it
to be null. Current observation schema v4 and support policy v2 make that field
optional and certificate-bound. Post-hoc re-evaluation under the repaired
contract gives 11 forward matches and one reverse match, hence
``unsupported``. That diagnostic does not rewrite or salvage the immutable
v5 artifact, whose archived result remains ``observer_failure``.

Even after intended parsing, “bent double-ended arrow” is overbroad. An
explicitly oracle-only post-hoc inspection of privileged Basic action programs
shows one precise nine-action positive template and a distinct near-miss
geometric program for the false-positive negative. The proposer and gate never
received those programs, and they cannot count as benchmark evidence. They
instead show why prose should specify a candidate while a frozen visual
contour/template or prototype scorer operationalizes it from pixels.

The subsequent v6 attempt selected the first lexicographic task in the
then-current live-eligible list,
``bd_advanced_lamp4-exist_quadrangle_five_lines12_0000_v6.json``, before
any of its pixels had been inspected. The selection is inspection-unbiased but
deterministic, not random. The file SHA-256 is
``6a120eabd4efeeee60b5555cbb581d6cced3d33206bb0ed556e61a29fb213057``.
Its support-release event
``sha256:b8fe3ea944d118058ac52e6f849ab5c1c1f6e08737f155e8b23f87569610877a``
advanced the ledger to its then-current ten-event head.

V6 ended ``proposal_error`` through
``plan_committed -> support_released -> proposal_failed``. It contains no
accepted proposal, support gate, query observation, or run archive. The
archived failure reports a blanket lexical rejection in category
``negative morphological complement``. This is a benchmark-attempt failure,
not a query score, and belongs in attempt coverage.

The old run schema did not archive the rejected raw proposer payload or its
receipt, so the lexical decision cannot be independently audited or replayed.
The next protocol revision must distinguish constructive morphological
descriptions from logical negation and persist rejected payloads and receipts.
A corrected future run would be a new attempt; v6 cannot be salvaged.

The v7 artifact, ``bd_arc_cup_0000_v7.json``, has file SHA-256
``9801dbec0928f59667993a993b99f2cfcd6d5c02264bb10ef467ac98c427a462``.
Its support-release event
``sha256:dbd578e1d3951837f25378721cf61e664eb96240e8f7c3fc108d1ff1db280a21``
produced successor
``sha256:fc82fcebf4686c36f85f9efa0944ef4fc57b5da41dfccb19126c33b372c146dc``.
V7 ended ``proposal_error`` because DNS resolution failed before a Codex
response arrived. It has no rejected proposal attempt, accepted proposal,
support gate, query observation, or run archive. This is a transport failure,
not a negative prediction or score.

The v8 artifact, ``bd_asymm_bridge_0000_v8.json``, has file SHA-256
``ef50e35732c9a02d933ca1d7628589071270b06bc3d87fd0bb2543cdff16ccdb``
and status ``complete``. Its phrase is “An enclosed region has a
glyph-decorated boundary.” The frozen operational rule requires an enclosed
cell whose boundary includes repeated small geometric glyphs such as circles,
squares, triangles, or zigzag teeth. All twelve support replays aligned and
both committed query predictions matched their revealed labels. The artifact
contains every phase from ``plan_committed`` through
``cold_replay_verified``, whose cold check covers all fourteen panel-byte
preimages. Its archive digest is
``4f679fe175383a3ceb85333bf85f644dbe2a1ab69033747ae4b7d133893dc2ef``
and its chain digest is
``c2cefb76126cc18d5f5b4e39c4b506fc259cb6fdb02ebf1a7dfa666f92631f4d``.
Event
``sha256:25317bb78b0cf60b7585f59c93c7331c0f6743c3553ae044008b14b69d76fd35``
produced intermediate twelve-event successor
``sha256:7cf70dcb4e15aa8f0d8f82f4e5ff1e32f3018fb1f467061a5c947b0a5cf742d3``.

V8 is one integration episode, not an accuracy estimate. Its two-query result
has the conditional 50% paired baseline described above. Canonical HYBRID also
remains an uncalibrated categorical self-observer: the same model family
proposes the phrase and judges panel matches. Replay establishes the archived
computation, not visual truth. A frozen, independently calibrated
pixels-to-score leg for open semantic claims and precise geometric near misses
is still the central theoretical hole.

The schema-v4 v9 artifact,
``hd_balanced_two-symmetric_transposed_0000_v9.json``, is an official-training
HD ordered-combination attempt with file SHA-256
``6171b6bca42ffa6423d0e7e1ef753da325ef3d000e6f39d2ca28b5afccf8e655``.
It proposed “A matched opposing pair of lobes joined at one center,” using the
cues ``paired_lobes``, ``matched_geometry``, ``central_junction``, and
``opposing_extents``. All twelve calls had one stable transport binding, but
the support gate rejected the rule: nine forward and three reverse matches,
seven ``present``, five ``nonmatch``, no errors, and no indeterminate outcomes.
One positive missed ``matched_geometry`` and two negatives were false
positives. Its phases stop at
``proposal_frozen -> support_gate_rejected`` with no query or run archive.
Event
``sha256:63983c4c918b23d8a009bca43a3390a1cf876bf96894521760761552dd8c11f8``
produced the current thirteen-event ledger head above.

V9 shows why vague visual correspondence needs a quantitative symmetry and
shape-matching leg. It also exposes an audit gap: a support-rejected schema-v4
run has no ``run_archive``, and the outer public plan stores only the
support-commitment digest rather than its nonce-bearing preimage. Current
public ``verify`` therefore cannot fully cold-bind and replay v9. A future
outer schema must persist the preimage on every post-proposal path; v9 does not
have v8's fourteen-preimage verification scope. Nothing here estimates
performance on the 1,800-task official test split.

Operational guide
-----------------

The canonical implementation surface is ``corpus.py``, ``release.py``,
``image_audit.py``, ``cohorts.py``, ``exposure.py``, ``evidence.py``,
``legs/contracts.py``, ``ir.py``, ``predicate_backend.py``, ``transport.py``,
``proposer.py``, ``synthesis.py``, ``artifacts.py``, ``benchmark.py``,
``admission.py``, and ``cli.py`` under ``bongard``.

The repository entry page is
`bongard/README.md <https://github.com/sashakolpakov/gkm/blob/master/bongard/README.md>`_.
See :doc:`reproduction` for validation and test commands and
``bongard/CONTINUATION_PLAN.md`` for the staged benchmark roadmap.
