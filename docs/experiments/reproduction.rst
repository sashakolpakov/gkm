Reproduction and Integrity Checks
=================================

Environment and tests
---------------------

Create a local environment and run the canonical Python tests:

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install -r bongard/requirements.txt
   .venv/bin/python -m pytest -q bongard/tests

The tests use small temporary fixtures; the 1.76 GB official archive is not
needed.  No Lean installation is required.  Python alone is authoritative for
predicate execution, calibration, replay, benchmark decisions, and scientific
artifact IDs.  A future Lean or other proof checker may only be a detached
optional sidecar over an already-frozen artifact; deleting it must change none
of those results, decisions, or IDs.

Verify the official corpus
--------------------------

The active experiment uses these external paths:

.. code-block:: text

   downloads/ShapeBongard_V2.zip
   downloads/ShapeBongard_V2_full/ShapeBongard_V2
   downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json

Verify the archive and extracted tree against the checked-in release
descriptor:

.. code-block:: bash

   .venv/bin/python -m bongard inventory \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --require-complete \
       --official-release \
       --archive downloads/ShapeBongard_V2.zip \
       --out results/bongard/official-inventory.json

The pinned identities are:

.. list-table::
   :header-rows: 1

   * - Object
     - SHA-256
   * - Archive
     - ``sha256:8c5542ac7b9ce8a6a14d157a0656dbde9da5b7843424eade4bd653759d9a27d0``
   * - Split file
     - ``sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230``
   * - Extracted corpus manifest
     - ``sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``
   * - Release descriptor
     - ``sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b``

Stage A1: historical failed command
-----------------------------------

The command below records the exact invocation of the 2026-08-06 Stage-A run.
It is shown for audit, not for literal re-execution: its seed has been consumed
and its artifact, exposure, and private-cache paths are write-once.  A new run
must use a newly generated 256-bit seed, new directories, and the latest
authentic exposure-ledger head.

The launcher was fingerprinted before the command.  The current launcher
digest is
``134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477``.

.. code-block:: bash

   .venv/bin/python -m bongard calibrate-semantic-stage-a \
       --corpus downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --archive downloads/ShapeBongard_V2.zip \
       --ledger-in downloads/ShapeBongard_V2_full/exposure/semantic_calibration_stage_a_v1/2b5c91500538f8b215732cbae460704cf8ef83ca6c9b46855c6785705b35d9d3.exposure.json \
       --expected-ledger-digest sha256:2b5c91500538f8b215732cbae460704cf8ef83ca6c9b46855c6785705b35d9d3 \
       --selection-seed f9ee0fc4433df603049734153ae5eeac7e7227873fd2f3f36bc163449f107857 \
       --selection-seed-provenance 'openssl-rand-hex-32@2026-08-06T09:02:42Z; generated-once-after-protocol-and-frame-freeze; no-rerolls' \
       --artifact-dir downloads/ShapeBongard_V2_full/semantic_calibration_stage_a_20260806 \
       --exposure-dir downloads/ShapeBongard_V2_full/exposure/semantic_calibration_stage_a_20260806 \
       --private-cache-dir downloads/ShapeBongard_V2_full/private/semantic_calibration_stage_a_20260806 \
       --expected-codex-launcher-sha256 134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477 \
       --candidate-count 48 \
       --model gpt-5.6-sol \
       --reasoning-effort medium \
       --proposer-minutes 15 \
       --scorer-minutes 10 \
       --workers 4

Before any panel or model access, the command durably wrote:

.. code-block:: text

   downloads/ShapeBongard_V2_full/exposure/semantic_calibration_stage_a_20260806/
       99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78.exposure.json

That file's canonical digest is
``sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78``.
The selection precommit is
``downloads/ShapeBongard_V2_full/semantic_calibration_stage_a_20260806/selection_seed_precommit.json``.
Its redundant ``full_corpus_manifest_digest`` field was manually mistyped as
``6fa515c673...``.  The file was left immutable.  A correction recorded during
the run is stored beside it as ``selection_seed_precommit_correction.json``
(file SHA-256
``5437a7ad6856f36a21ad88117f3d0676c65537e6efb6f8a7e2da9eb62766ff98``).
The authoritative predecessor ledger, release descriptor, command, and
successor ledger all bind the correct manifest
``sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``.

The terminal fields are:

.. code-block:: text

   stage_a_result = failed
   command_receipt_digest = 9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb
   terminal_failure_digest = a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83
   fitted_campaign = none

A1 completed all 48 proposer calls: 37 emitted accepted soft claims, 10 were
direct-only attrition, and one was rejected by the typed parser.  All 37 scorer
calls were transport errors.  There were zero scores, labels remained withheld,
and no bins, intervals, fitted calibration, semantic accuracy, or negation
evidence were produced.  Its frozen scorer schema used provider-incompatible
``minItems``, ``maxItems``, and ``uniqueItems`` keywords.

Stage A2: invalidated distinct experiment
-----------------------------------------

The repaired transport schema removes those unsupported keywords; Python still
enforces exact cue coverage, order, uniqueness, and witness ownership.  This
changes the protocol identity.  A2 is a new experiment with a fresh selection,
not a retry of A1:

.. code-block:: text

   protocol_digest = 2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca
   predecessor_ledger = sha256:99597cf6477cd7e145c3bf62daf885fe7bf5ef5c0c829741353b5d6a0f5d7a78
   selection_seed = eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1
   successor_ledger = sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8
   state = invalidated-by-live-source-mutation; no Stage-A terminal artifact

The A2 seed precommit is
``downloads/ShapeBongard_V2_full/semantic_calibration_stage_a2_20260806/selection_seed_precommit.json``.
Its durable successor is
``downloads/ShapeBongard_V2_full/exposure/semantic_calibration_stage_a2_20260806/9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8.exposure.json``.
A concurrent agent edited ``bongard/typed_visual_proposal.py`` after the A2
protocol and cohort were frozen.  The live grammar digest then differed from
the frozen protocol, and the process exited without writing a Stage-A terminal
artifact.  The incident record is
``downloads/ShapeBongard_V2_full/semantic_calibration_stage_a2_20260806/A2_INVALIDATED_SOURCE_MUTATION_INCIDENT.json``
with file digest
``sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4``.
Process output showed 48 proposer and 34 scorer launches only; their outputs
were lost.  Labels were not revealed, no calibration, accuracy, or semantic
inference is valid, and the same cohort may not be rerun.

Stage A3: terminal scientific failure
-------------------------------------

A3 ran the production headless-Codex path from the frozen source frame.  Its
selection and artifact locations are write-once and its 22 selected tasks are
exposed; the following addresses are for audit, not re-execution:

.. code-block:: text

   command_receipt = downloads/ShapeBongard_V2_full/semantic_calibration_stage_a3_20260806/artifacts/2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681.stage-a-command-receipt.json
   command_receipt_digest = sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681
   terminal_failure = downloads/ShapeBongard_V2_full/semantic_calibration_stage_a3_20260806/artifacts/cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb.semantic-calibration-failure.json
   terminal_failure_digest = sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb
   process_exit = 2
   reason = calibration score bins are underpopulated: 1

All 22 proposer calls succeeded.  Fifteen records contained accepted soft
claims, six were direct-only attrition, and one was rejected by the typed
parser.  All 15 resulting scorer calls succeeded.  Their ordinal distribution
was eight ``0``, one ``0.5``, and six ``1``.

.. list-table::
   :header-rows: 1

   * - Fixed score bin
     - Clusters
     - Affirmative labels
   * - ``[0, 0.75)``
     - 9
     - 1
   * - ``[0.75, 1]``
     - 6
     - 5

The preregistered minimum was eight clusters in each bin, so no calibration
was fitted and Stage B was not authorized.  Intended-bin orientation was
13/15 versus 2/15 for its exact complement; at the naive ``score >= 0.5``
threshold it was 12/15 versus 3/15.  Negation did not win.  A3 consumed 22
tasks and leaves 10,047 exact-unused train/validation task IDs (FF 2,998, BD
3,434, HD 3,615).  SEALED/test remained untouched.

The typed rejection was caused by a parser expression matching the prefix
``def`` in the ordinary word ``defines``.  The parser now requires a complete
forbidden-keyword match.  This post-A3 fix does not change the A3 artifact.

Do not overread A3's ``codex_launcher_digest``.  It is the digest of the
installed JavaScript wrapper, which dynamically spawned a separate native
client.  The receipt binds that wrapper and reported version
``codex-cli 0.146.0``; it does not authenticate the native bytes.  The current
native file's post-hoc digest cannot repair that historical gap.  A new live
command must resolve and execute the native binary directly and bind its exact
digest before and after every call.

Stage B: unauthorized and capacity-limited
-------------------------------------------

No canonical Stage-B CLI reproduction command is published.  Do not improvise
one from internal Python APIs.

Neither the failed A1 receipt, the invalidated A2 incident, nor the failed A3
fit can authorize Stage B.  No Stage-B run is currently declared.  The completed metadata-only
audit gives remaining DRILL maximum 24 BD + 0 constituent-disjoint HD = 24 and
full-ledger-disjoint DEV capacity 16 BD + 0 HD = 16.  The default request for
24 fails before pixels/exposure/model; a 16-task BD-only pilot remains below
the frozen 24-cluster minimum and cannot authorize SEALED execution.

The earlier 28-unit upper bound applied HD constituent disjointness only inside
the proposed new batch; it did not initialize the exclusion set from every
complete-A2 exposure.  Production selection v2 performs that projection, and
every remaining DRILL HD pair shares a constituent with it.

Those DRILL figures describe the reservoir before A3.  Running the exact v3
selector against successor ledger
``sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4``
certifies post-A3 strict DRILL capacity zero, with zero eligible tasks and
groups.  Its certificate is
``sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1``.
The corresponding DEV capacity remains 16 BD + 0 HD, certificate
``sha256:434c0756e89891c4a10e31fdf0c97e2e9373930a2ed48e1ecfa011c36f15c4c8``.

Do not confuse the strict 24-unit frame with the archive inventory.  After A3,
10,047 of 10,200 train/validation task IDs remain exact-unused: 2,998 FF,
3,434 BD, and 3,615 HD.  The current selector deliberately
collapses or excludes shared semantic constituents.  A future expanded
calibration frame may use exact-unused training tasks only with a newly frozen
selection policy and explicit dependence accounting; these counts are not
permission to weaken DEV or SEALED isolation.

A3 fixed ``minimum_clusters_per_bin = 8`` before its seed and model output.
With two bins, 90% confidence, and the fixed 0.5 boundary, its simultaneous
Hoeffding radius was 0.480161.  The upper bin reached only six.  Future
calibration must recruit label-blind through a frozen order until bins are
powered, or preregister a fixed batch sized for measured proposal attrition and
score-bin occupancy.

New Stage-A command receipts use schema v2 and bind the complete authoritative
Python source snapshot.  The exact non-authoritative
``bongard/semantic_checker.py`` proof-checker sidecar is excluded: installing,
editing, or deleting it cannot change a receipt identity, while a mutation to
any potentially authoritative Python module remains fail-closed.  Historical
failed v1 receipts remain exactly auditable but cannot authorize Stage B.  Any
post-precommit source mutation now produces a durable typed operational failure
with labels withheld.

SEALED is unavailable
---------------------

There is no visual-semantic official-test command.  ``--sealed-test`` with
``--predicate-mode visual-semantic`` is rejected before corpus/panel release,
and the benchmark API independently rejects the same official-test execution.
Do not substitute the generic single-episode command or call an internal
runner to bypass this control.

What cold replay establishes
----------------------------

Successful artifacts bind exact panel bytes, typed witness bundles, the
registry snapshot, the closed Python formula, calibration campaign, support
gate, query commitments, predictions, reveal, launcher fingerprint, and cache
binding.  Cold replay reconstructs and checks those objects without a model.

The atomic smoke has a narrower replay contract.  It binds 12 neutral support
descriptions, one text-only atom proposal, the complete support atom matrix,
the frozen positive conjunction, two query descriptions and observations, and
the durable prediction-before-label boundary.  A successful run contains
exactly 29 distinct causal receipts.  ``operational_nonmatch`` remains a
distinct operational record and projects to semantic ``indeterminate``; replay
cannot upgrade it to certified absence.

It establishes that the recorded computation is internally consistent and
tamper-evident.  It does not prove that a model's phrase such as ``bird-like``
is perceptually correct, that public images were absent from model pretraining,
or that any future exploratory DRILL/DEV accuracy generalizes to SEALED tasks.

What A3 says about the implementation
--------------------------------------

A3 validates transport, not the synthesis architecture.  The proposer records
rich descriptions of all twelve panels, but those descriptions are audit-only.
It makes one irreversible guess from zero to three direct catalog atoms and at
most one soft claim bundling one to four cues.  Python synthesis only lowers
that guess to a conjunction; it does not derive atomic facts or search
candidates.  The scorer collapses all cues for a panel by minimum into
``0``, ``0.5``, or ``1``.

That reproducible unit is now implemented in
``atomic_semantic_synthesis.py`` and ``atomic_smoke_runner.py``: one-phrase
atoms, complete atom-by-panel observations, deterministic positive
conjunctions, no ``Not``, and no polarity flip.  The proposer is causally
restricted to the frozen support descriptions.  The remaining perception gap
is richer typed object/part/angle/relation grounding; a one-sentence vision
description is still lossy.  Lean or another checker may inspect a frozen
artifact only as an optional removable sidecar; it is not part of the
predicate, result, or ID.

Exploratory atomic smoke
------------------------

The production command authenticates the complete 12,000-task release, exact
split, complete manifest, and A3 successor ledger before selection.  It
persists a secret-free command configuration before generating any selection,
episode, or label-seal secret.  It then durably records the exact selected-task
exposure before hashing a selected panel, stages the pinned native Codex bytes,
and guards the authoritative source before and after every call.

Run it only from a committed immutable checkout.  Every store argument must
name an already existing absolute, non-symlink directory with mode ``0700``:

.. code-block:: bash

   PYTHONPYCACHEPREFIX=/absolute/empty/pycache \
   python -B -m bongard.atomic_smoke_command \
     --corpus /absolute/path/to/ShapeBongard_V2 \
     --archive /absolute/path/to/ShapeBongard_V2.zip \
     --exposure-ledger /absolute/path/to/a3-successor.exposure.json \
     --config-store /absolute/path/to/config-store \
     --exposure-store /absolute/path/to/exposure-store \
     --prediction-store /absolute/path/to/prediction-store \
     --terminal-store /absolute/path/to/terminal-store \
     --cache-store /absolute/path/to/cache-store

The command prints one selected-ID-redacted JSON line.  Selection is from the
ten exact-unseen training tasks whose Basic-shape generator is already exposed.
Consequently this is an exploratory transport/synthesis smoke, not an
independent calibration, DEV estimate, or official benchmark.  All four
authorization flags remain false regardless of its two-query score.

First live N=1 incident
-------------------------------

The first live invocation from commit
``62ea577f5d86d109577f4f5e49b8b4866eb76c92`` and tag
``bongard-atomic-pre-smoke-20260806`` persisted cache content address
``sha256:1094dfd6794d4dfd141b9d0d1c89cf648d5c7d57ea0a545868bc38df928f28a4``,
config address
``sha256:9dad0a5f468d1e8f3c65f7b83ac1ce7d2072e6541078bfbe9b4289ae3abdd451``,
and exposure-successor address
``sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a``.
It persisted no prediction and no terminal.  The exact selected task is
therefore consumed and must not be rerolled.

The CLI reported ``AtomicSmokeCommandError`` with exact message ``failed run
precommit is not canonical JSON`` and reason digest
``2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d``.
The runner was entered and returned a typed ``AtomicSmokeRun``.  Fallback
terminal construction then tried to JSON-clone its frozen ``MappingProxy``
precommit.  Normal terminal construction contains the same deterministic
defect, but the surviving outer error does not establish which exception first
entered fallback.  Status, phase, output, and successful model-call count are
irrecoverable; the only valid count is unknown in the inclusive range 0--29.
With no persisted prediction, labels could not be materialized or revealed.
This operational failure supplies no score and authorizes no calibration,
semantic, benchmark, or official-test claim.

The sanitized machine record is
``bongard/data/atomic_smoke_n1_operational_failure_v1.json``.  A preceding
setup invocation had rejected a cache store at mode ``0755`` instead of the
required ``0700``.  That setup invocation persisted no exposure and consumed
nothing.
