Reproduction and Integrity Checks
=================================

Environment
-----------

Create a local environment and install the declared Bongard dependencies:

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install -r bongard/requirements.txt

Run the canonical package tests:

.. code-block:: bash

   .venv/bin/python -m pytest -q bongard/tests

These tests do not require the official image archive. They construct small
temporary corpus fixtures and check corpus validation, exposure integrity,
four-disposition evidence, typed calls, signed ``AffirmativeRelation``
contracts, rejection of undeclared inequality directions, the closed IR,
artifact tampering, cold replay, admission gates, and atomic promotion.
They also require no Lean installation. Pure Python is the reference predicate
backend; any future Lean checker is an optional conformance check whose removal
must leave artifact identities, admission, and cold replay unchanged.

Validate the exact official corpus
----------------------------------

The full archive is external data and must not be committed to this
repository. Validate it and record the resulting content address:

.. code-block:: bash

   python3 -m bongard inventory \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --require-complete \
       --official-release \
       --archive downloads/ShapeBongard_V2.zip \
       --out results/bongard/official-inventory.json

Expected counts are 12,000 total, split 9,300/900/1,800. The official test
regimes must be 600 ``FF``, 480 ``BA``, 400 ``CM``, and 320 ``NV``. The
validator also checks seven positive and seven negative PNGs per task, PNG
signatures, family placement, split coverage/disjointness, and every content
hash against the checked-in release descriptor. The pinned extracted-corpus
digest is
``sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``.

The full decoder audit was also run twice. All 168,000 panels are single-frame
512 by 512 RGB PNGs with no Pillow info keys; zero anomalies were observed.
The strict report is ``bongard/data/shape_bongard_v2_image_audit_v1.json`` and
has digest
``sha256:d3485ada3605d708db82fbcfe6ecfc73506ce51ed85fcd1ce6ccd798e3bff9f8``.

Freeze exposure state before adaptation
---------------------------------------

Import all known historical disclosures before drawing a new partition:

.. code-block:: python

   from bongard import (
       ExposureLedger,
       ShapeBongardCorpus,
       deterministic_partition,
       import_historical_exposures,
       load_historical_exposure,
   )

   corpus = ShapeBongardCorpus.discover(
       "downloads/ShapeBongard_V2_full/ShapeBongard_V2",
       split_file="downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json",
       require_complete=True,
   )
   manifest = corpus.build_manifest()
   historical = load_historical_exposure()
   ledger = ExposureLedger.create(manifest.digest)
   ledger = import_historical_exposures(
       ledger,
       historical.exact_official_task_ids,
       source="bongard/data/historical_exposure_v1.json",
       known_task_ids=corpus.task_ids,
   )
   eligible = ledger.unseen_task_ids(corpus.task_ids)
   partition = deterministic_partition(
       eligible,
       drill_count=300,
       dev_count=75,
       namespace="bongard-unused-v1",
   )
   print(ledger.digest)
   print(partition.digest)
   ledger.write_content_addressed("results/bongard/exposure")

This exact-task partition is intentionally not the semantic eligibility
decision. Use ``python3 -m bongard cohorts`` to select only tasks whose Basic
family or Abstract attribute pair is also historically clean. Persist the
ledger and any selected cohort digest before an adaptive proposer sees drill
panels. Development code must reject the sealed IDs. For HD, this historical
qualification means only that the ordered attribute combination is absent from
the bound history. Its component attributes may be familiar, and all 20
generated instances of the pair are one semantic sibling group.

The frozen drill membership contains 2,769 tasks, but live availability must be
recomputed with ``--ledger-in``. At the checked-in sixteen-event campaign head
``sha256:da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a``,
``python3 -m bongard cohorts --cohort drill --ledger-in ...`` reports 1,744
resolver-v2 live tasks and 1,025 semantic exclusions, including sixteen
exact-task collisions. The ledger records 22 tasks and 38 semantic keys,
producing 199 effective exposed keys after policy blocking. The overlay digest
is
``sha256:9e7ad95bc0fe2200d647c7ef9c34b81f8b041115265175be6fe63d6c67562dde``;
its live-membership digest is
``sha256:be680542b28a855d54cedcda6726d140af1ce4a8ad97c008511d5843f4e4b7e1``.
The checked-in ledger file is
``downloads/ShapeBongard_V2_full/exposure/abstract_006/da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a.exposure.json``.

Resolver v1's earlier 2,609 count was an overclaim because it did not group
numbered Basic morphology siblings. Resolver v2 removes a terminal number or
``_newN`` for a conservative sibling key and blocks clusters crossing frozen
cohort boundaries. Its policy digest is
``sha256:48598ae580a2f88aee7652d36fd386d54a8e4265b040bf1313f558508f47af9a``.
Before v7–v9, its initial training intersection contained 1,290 tasks
across 161 groups. At the current head, ``--split train --cohort drill`` has a
2,096-task historical scope, 1,238 live tasks, and 858 exclusions, including
sixteen exact-task collisions; its overlay
digest is
``sha256:64c7f3cbd4444829d1bd8c50d1a99cc95d5830ec6459879a5a7f6668868eee90``;
its live-membership digest is
``sha256:2619ea03a9f32bddef941818791fee9d477040f073043da2c715547474813a23``.
Resolver collision domains are not statistically independent samples. The
retrospective montage event and v6–v12 support releases filter their related
groups from subsequent runs. The overlay is not a new frozen partition and
does not certify unseen bytes. Run-time ``external_anchor`` fields are null;
committing the ledger later is only after-the-fact publication, not
preregistration or proof that it was the globally latest head.

Canonical episode
-----------------

The episode API is in ``bongard.benchmark``:

.. code-block:: python

   from bongard.benchmark import SupportGatePolicy, prepare_episode, run_episode

   task_id = corpus.tasks_in_split("val")[0].task_id
   plan = prepare_episode(
       corpus,
       task_id,
       seed="development-seed-0",
       corpus_manifest=manifest,
   )
   result = run_episode(
       plan,
       proposer,
       observer,
       support_gate_policy=SupportGatePolicy.empirical(),
   )
   print(result.score.to_data())

``prepare_episode`` chooses six support panels per side and one query panel per
side without exposing their labels through the query view. ``run_episode``
makes one proposer call, fixes the claim/formula/registry, and then makes twelve
fresh neutral single-image support observations. All twelve must align before
the final proposal freeze binds the support-gate digest and releases the two
queries. It commits both query predictions before revealing labels for scoring.
A successful headless episode therefore makes fifteen model calls: one
proposal, twelve support replays, and two query observations. A sealed run
additionally supplies a ``SealedTestGuard`` whose capture is verified after
execution.

The proposer and observer are protocols rather than trusted classifiers. They
receive staged panel bytes and neutral temporary paths; they must not receive
source paths, source filenames, task IDs, split/regime metadata, or labels
outside the support view.

Reproduce the new measurement cores
-----------------------------------

The focused deterministic checks are:

.. code-block:: bash

   .venv/bin/python -m pytest -q \
       bongard/tests/test_bilateral_symmetry.py \
       bongard/tests/test_support_prototypes.py

``bongard.legs.bilateral_symmetry`` consumes only panel bytes, uses fixed
thresholds and a fixed reflection-axis grid, returns all four evidence
dispositions, and registers only ``AT_LEAST``. Its interval is a deterministic
preprocessing-sensitivity envelope, not external calibration. The post-hoc v9
audit is deliberately a failed scientific check: several negatives score as
more symmetric than positives. The next geometry extractor must expose
part/lobe ownership, junctions, and part correspondence rather than merely tune
this score.

``bongard.legs.neutral_features`` performs deterministic panel-only extraction
of eighteen interval features. ``bongard.prototype_episode`` freezes all 6+6
support extractions before one Codex catalog-selection call, fits the selected
positive/negative prototypes, re-extracts all support panels, and uses the
generic runner's freeze/query/reveal sequence. Query evaluation is pure Python.
The fixed orientation is
``distance(query, negative) - distance(query, positive)`` and has no
polarity-flip operation.

The canonical development calibration keeps all twelve tasks in the
denominator and obtains 0/12 strict support passes for every group; its best
query result is 10/24 images and 2/12 puzzles. Do not cite this as a benchmark
score. Component cold replay verifies archived receipts, feature packets,
fitting, formula, and evidence. The public whole-run PURE verifier additionally
reruns pixel extraction from every exact PNG byte preimage, refits the frozen
prototypes, recompiles the Python IR, and checks support and any released query
evidence without Codex or Lean.

Cold replay
-----------

``bongard.artifacts`` stores canonical JSON for support, proposal freeze,
query release, atom evidence, predictions, and label reveal. To check only the
logical result in a process with no model or registered implementation, load
the frozen formula data and cold inputs and call:

.. code-block:: python

   from bongard.artifacts import replay_cold_payload

   replayed = replay_cold_payload(formula_data, cold_inputs_data)

The formula digest, all atom paths, and every provenance-bearing disposition
are checked. Full bundle verification also checks the parent digest chain,
support/query byte disjointness, committed predictions, and label coverage.
The stricter completed-run verifier additionally reparses the proposal and
canonically recompiles the verifier-owned HYBRID formula, registry snapshot,
source and operational identities, and attachment contract without invoking
the model.

Promotion audit
---------------

An accepted leg requires typed attachment; a match between stripped submitted
source and normalized implementation source (or its selected bytecode
fallback), plus a direct formula call; a verifier-precommitted incumbent;
nuisance stability; calibration versus the fixed baseline; near-miss
contrasts; anti-memorization; candidate replay; complete archive replay; and
AST novelty. Applying the decision is compare-and-swap against the exact
archive digest inspected during admission. Rejected decisions contain no next
archive. Receipt DTOs are not cryptographic signatures: production receipts
must be constructed inside the verifier-owned process or authenticated
externally.

Canonical command line
----------------------

Inventory and content-address the complete corpus:

.. code-block:: bash

   python3 -m bongard inventory \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --require-complete \
       --out results/bongard/corpus-inventory.json

Run one explicitly selected, historically clean training drill with a headless
Codex proposer. ``--cohort drill`` makes the official CLI check membership in
the frozen semantic drill cohort. ``--ledger-in`` must name the externally
maintained latest content-addressed ledger, and the runner writes its one-event
successor before releasing support pixels. The CLI cannot prove by itself that
the supplied predecessor is the latest authentic ledger head.
Official runs also require an externally recorded SHA-256 of the fixed
``codex`` launcher; this prevents the run command from substituting an
arbitrary executable. Inspect and record the fingerprint before support is
released, then record the run-file SHA-256 outside the run and cold-verify
without calling the model:

.. code-block:: bash

   python3 -c 'from bongard.transport import codex_cli_fingerprint; print(codex_cli_fingerprint())'
   python3 -m bongard run \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --task-id HISTORICALLY_CLEAN_TRAIN_TASK_ID \
       --seed UNIQUE_PRECOMMITTED_SEED \
       --out results/bongard/drill.json \
       --exposure-dir results/bongard/exposure \
       --ledger-in results/bongard/exposure/PRIOR_DIGEST.exposure.json \
       --require-unseen \
       --cohort drill \
       --predicate-mode support-prototype \
       --prototype-calibration bongard/data/support_prototype_calibration_v1.json \
       --official-release \
       --archive downloads/ShapeBongard_V2.zip \
       --expected-codex-launcher-sha256 EXTERNALLY_RECORDED_CODEX_SHA256
   shasum -a 256 results/bongard/drill.json
   python3 -m bongard verify \
       --run results/bongard/drill.json \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --archive downloads/ShapeBongard_V2.zip \
       --expected-sha256 EXTERNALLY_RECORDED_RUN_SHA256

Output and ledger paths are write-once. Official test tasks additionally
require ``--sealed-test`` and a complete-corpus guard; drill and development
commands must not use that flag. A proposal failure has no replayable archive
and fails verification. It is an infrastructure result, not a negative
prediction and not a benchmark score.

The checked-in v3 drill attempt
``bongard/runs/official_complete_drill_20260805/hd_exist_quadrangle-exist_sector_0000_v4.json``
(file SHA-256
``bf60a36bc7a48e61c61c8de2153753fa2996db54eacf53fe4c861bf06a9b4f41``)
is an example. It ended ``support_rejected`` before query release and has
no query observations or run archive. All twelve support transports returned
raw judgments, fitting 10/12 labels: six of six positives were ``present``;
four of six negatives were ``nonmatch`` and two were ``present``. The
verifier nevertheless converted all twelve to ``TransportIdentityError``
because ``cloud_config_bundle_cache_binding`` varied across calls. The
artifact therefore diagnoses a transport-identity protocol bug, not query
accuracy.

The later checked-in attempt
``bongard/runs/official_complete_drill_20260805/bd_advanced_lamp3_0000_v5.json``
has file SHA-256
``e3fbe8f76290bb93f33def26c36b50f9ae451e43456a52e4796976a71662255a``.
All proposal and support receipts bind the same cache value,
``sha256:6860e08631caee1357061bd727e93f7d200931b3bb2d925f873aea3d669d22f2``.
The run still ended ``support_rejected`` before query release, with no query
observations or run archive. Its immutable v3 observation/v1 gate records six
forward matches, one reverse match, and five parser errors. The raw outputs
were all six positives ``present``; among negatives, five were
``nonmatch`` and one was ``present``.

Those five nonmatches exposed a contract mismatch: their top-level
``reason`` was prompt/schema-allowed or inferred, but the archived parser
required null. The current observation schema v4 and support policy v2 make a
nonmatch reason optional and certificate-bound; all 418 package tests pass.
Re-evaluating only the archived raw payloads under that repaired parser gives
11 forward and one reverse match, whose gate result is ``unsupported``.
This post-hoc diagnostic does not alter the v5 file: its archived result remains
``observer_failure``, and it cannot be salvaged into a completed episode or
query score.

The remaining reverse match is perceptual, not a parser artifact. The phrase
“bent double-ended arrow” also matches a negative near miss. A post-hoc view of
the Basic action programs shows a precise nine-action positive template and a
distinct near-miss geometric program for that negative, but this is privileged
oracle metadata. It was unavailable to the proposer and support gate and must
not enter a visual benchmark run. The actionable reproduction target is a
frozen contour/template or prototype scorer that operationalizes the prose
claim from panel pixels.

The upstream sampler definitions provide a broader post-hoc oracle diagnosis:
Basic multi-shape and Abstract attribute-pair positives are conjunctions, and
their negatives may be different near-miss subgroups that fail different
conjuncts. The proposer, support observer, synthesis path, and predictor never
received task IDs or action programs. The current proposer prompt therefore
requires all-positive cue coverage and requires each negative to fail at least
one cue while preserving the distinct failed conjuncts. This prompt constraint
does not authorize action-program metadata or turn prose into calibrated
evidence.

The next attempt chose the first lexicographic entry in the then-current
live-eligible list, without prior inspection of that task's pixels:
``bongard/runs/official_complete_drill_20260805/bd_advanced_lamp4-exist_quadrangle_five_lines12_0000_v6.json``.
This deterministic selection avoids inspection-based cherry-picking but is not
random sampling. The file SHA-256 is
``6a120eabd4efeeee60b5555cbb581d6cced3d33206bb0ed556e61a29fb213057``.
Its support-release event
``sha256:b8fe3ea944d118058ac52e6f849ab5c1c1f6e08737f155e8b23f87569610877a``
produced its then-current ten-event ledger head.

V6 has status ``proposal_error`` and phases ``plan_committed``,
``support_released``, and ``proposal_failed``. It has no accepted proposal,
support gate, query observations, or run archive. Its failure reason records
the blanket lexical category ``negative morphological complement``. This is
a benchmark-attempt failure and must not be converted into a query score.

The old artifact schema omitted the rejected raw payload and proposer receipt.
Consequently the lexical rejection cannot be independently inspected or
replayed from the run file. The required repair is to distinguish constructive
morphological wording from logical negation and persist every rejected payload
and receipt. Re-running after that repair creates a new attempt; it does not
salvage v6.

The next artifact is
``bongard/runs/official_complete_drill_20260805/bd_arc_cup_0000_v7.json``
with file SHA-256
``9801dbec0928f59667993a993b99f2cfcd6d5c02264bb10ef467ac98c427a462``.
Its event
``sha256:dbd578e1d3951837f25378721cf61e664eb96240e8f7c3fc108d1ff1db280a21``
produced successor
``sha256:fc82fcebf4686c36f85f9efa0944ef4fc57b5da41dfccb19126c33b372c146dc``.
V7 ended ``proposal_error`` because DNS resolution failed before a Codex
response. It has no rejected proposal attempt, accepted proposal, support gate,
query observation, or archive, so it is a transport failure rather than a
score.

The completed artifact is
``bongard/runs/official_complete_drill_20260805/bd_asymm_bridge_0000_v8.json``
with file SHA-256
``ef50e35732c9a02d933ca1d7628589071270b06bc3d87fd0bb2543cdff16ccdb``.
It proposes “An enclosed region has a glyph-decorated boundary” and freezes the
operational rule that an enclosed cell has a boundary segment rendered as
repeated small circles, squares, triangles, or zigzag teeth. The support gate
aligned 12/12; both query predictions matched their revealed labels; and every
phase from ``plan_committed`` through ``cold_replay_verified`` is present. Cold
verification checks all 14 official panel-byte preimages. The archive digest is
``4f679fe175383a3ceb85333bf85f644dbe2a1ab69033747ae4b7d133893dc2ef``
and the chain digest is
``c2cefb76126cc18d5f5b4e39c4b506fc259cb6fdb02ebf1a7dfa666f92631f4d``.
Its event
``sha256:25317bb78b0cf60b7585f59c93c7331c0f6743c3553ae044008b14b69d76fd35``
produced intermediate twelve-event ledger head
``sha256:7cf70dcb4e15aa8f0d8f82f4e5ff1e32f3018fb1f467061a5c947b0a5cf742d3``.

Do not turn v8's 2/2 into a benchmark estimate. It is one integration episode
with the conditional 50% paired baseline. HYBRID remains an uncalibrated
categorical self-observer, so cold replay proves that the archived judgments
were used consistently, not that they are true of the pixels. The next
reproduction target is the still-missing frozen, independently calibrated
pixels-to-score leg for soft semantics and precise geometric near misses.

The schema-v4 artifact
``bongard/runs/official_complete_drill_20260805/hd_balanced_two-symmetric_transposed_0000_v9.json``
is an official-training HD ordered-combination attempt with file SHA-256
``6171b6bca42ffa6423d0e7e1ef753da325ef3d000e6f39d2ca28b5afccf8e655``.
It proposes “A matched opposing pair of lobes joined at one center,” with the
cues ``paired_lobes``, ``matched_geometry``, ``central_junction``, and
``opposing_extents``. All twelve calls used a stable transport binding. The
support gate nevertheless rejected it with nine forward and three reverse
matches: seven ``present``, five ``nonmatch``, no errors, and no indeterminate
outcomes. One positive missed ``matched_geometry`` and two negatives were false
positives. Its phases end at
``proposal_frozen -> support_gate_rejected``, before query release or a run
archive. Event
``sha256:63983c4c918b23d8a009bca43a3390a1cf876bf96894521760761552dd8c11f8``
produced intermediate thirteen-event ledger head
``sha256:65c8dd508f6c21e64b0c777a83159a470fbab12cfb8fee6adf588c0a9c400c8b``.

Public verification cannot fully cold-bind v9. A support-rejected schema-v4
artifact has no ``run_archive``, and its outer plan stores only the
support-commitment digest, not the nonce-bearing commitment preimage. Reproduce
this as an expected audit limitation, not as v8-style fourteen-preimage replay.
Outer schema v5 retains that preimage on every new exit and verifies an ordinary
support rejection against all twelve exact support PNGs. It cannot retroactively
repair v9. Scientifically, the positive miss and two false positives make a
quantitative symmetry/shape-matching leg the next target.

The schema-v5 artifact
``bongard/runs/official_complete_drill_20260805/bd_asymm_trap_bridge-trans_arc_cup_0000_v10.json``
has file SHA-256
``0bdf82438b3b85b368f0c0fb93298f184fbae55b0b5777c06759670b53c3b8a7``.
V10 ended ``proposal_error`` because sandbox DNS failed before a Codex
response. Its support-commitment preimage is present, but there is no validated
response, proposal receipt, support gate, query observation, or archive. Event
``sha256:dee13f7dae4e949882f516b8e8ca54eec7af8db0aa1fc47ca8a90aadb50195d7``
produced successor
``sha256:1a547a92e7897558e2f5f3e209545309d1f2ec41b4650d7b724ab4193840eff7``.
This reproduces as infrastructure failure, not a class outcome.

The v11 artifact
``bongard/runs/official_complete_drill_20260805/bd_asymm_unbala_goldfish-asymmetric_crown_0000_v11.json``
has file SHA-256
``0a324a7fc780dea392443a9afd54dbfe19fe5631d06ff1287abba7da342ac561``.
Its phrase is “Complete paired motifs with shared rotational handedness.” The
support gate rejected it with eight forward and four reverse matches: ten
``present``, two ``nonmatch``, no errors, and no indeterminate outcomes. Public
verification successfully reproduces all twelve exact official support
preimages. Event
``sha256:395fbacc33c3bc206a581e2d85cf856b89e978ce6133a3a2574e193d6d7484ab``
produced successor
``sha256:8841cac62c203a2895a176c2cfbef8b97b46cfb33a6e0db2072c24efb54dc171``.

The v12 artifact
``bongard/runs/official_complete_drill_20260805/hd_closed_shape-has_obtuse_angle_0000_v12.json``
has file SHA-256
``e7c62b4eb96e910d5ea2738fb6622ab9b469993befc7e0897906f5ed223960df``.
Its phrase is “A cyclic enclosure with an inward re-entrant feature.” The gate
rejected it with seven forward and five reverse matches: nine ``present``,
three ``nonmatch``, no errors, and no indeterminate outcomes. Public
verification again reproduces all twelve exact support preimages. Event
``sha256:ce8f67fc54e3775932951c622d9f87dac805a12ac082bc66f5bc258764492c2e``
produced the current sixteen-event ledger head above.

Treat v10–v12 as one three-attempt smoke campaign: proposer success 2/3,
support-gate pass 0/2, query release 0/3, and completion 0/3. Its 26 successful
receipts report 233,921 known input tokens, 23,552 cached input tokens, 15,001
output tokens, and 10,694 reasoning tokens. No query pixels were released, so
there is no query accuracy to reproduce or report.

Twelve-task PURE drill campaign
-------------------------------

The completed headless PURE campaign is recorded in
``bongard/data/support_prototype_drill_result_v1.json``. Its canonical content
digest is
``sha256:38a89b3f78afa7c89f2f9dc881d209fce7b791ef3a346e54ee9ee3abaffa7fca``.
All twelve raw records passed the public exact-official verifier, totaling 276
fresh neutral-extraction replays and no missing panel preimages. The canonical
verification summary is
``bongard/data/support_prototype_drill_verification_v1.json``, content digest
``sha256:2fdfd965916450cf4165201464e68369dd523ed44806caa5994e7e5ddaa07729``.
All twelve preregistered attempts remain in the denominator: zero completed,
eleven ended ``support_rejected``, and one ended in a replayable proposal-parser
error. The support gate therefore passed 0/12. No query pixels were released,
so the result contains neither a query score nor an accuracy estimate.

The eleven parsed proposals produced 132 executable support labels. Of those,
46 were correct, 10 were wrong, and 76 were indeterminate. This distribution is
the concrete failure, not a hidden query result. Codex prose often articulated
object-level relations between shapes, parts, and orientations, while the
bridge compiled the proposal to exactly one precommitted interval-centroid
feature group. The executable predicate consequently discarded much of the
information present in the description. The twelfth attempt exposed an
independent parser bug: descriptive asymmetry language was falsely classified
as logical negation. That parser error is replayable and is not a negative
prediction. The parser now admits constructive component relations such as
``separated`` and ``disconnected`` while the closed formula grammar continues
to forbid logical ``Not`` and polarity reversal.

The campaign therefore falsifies the current one-group prototype bridge before
query release; it does not show that headless vision supplied no useful
description. Predicate serialization, evaluation, and whole-run replay use the
canonical pure-Python backend. Lean was not required and remains optional and
removable.

Completed-run verification recomputes each Codex receipt's internal hashes and
request/response bindings. The receipt is not a provider signature and may
record that Codex JSONL omitted the reported model; launcher pinning does not
authenticate remote model execution.

Pre-rewrite mechanism controls and exploratory pilots are preserved at the
annotated Git tag ``pre-bongard-complete-rewrite-20260805``. Stale working-tree
copies remain physically present pending explicit deletion, but they are
excluded from the canonical reproduction path.

Build these documents
---------------------

.. code-block:: bash

   .venv/bin/python -m pip install -r docs/requirements.txt
   .venv/bin/sphinx-build -b html docs docs/_build/html

The normal Sphinx configuration also regenerates ARC artifact pages. Use a
clean worktree or review those generated changes separately.
