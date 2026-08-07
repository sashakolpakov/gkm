Reproduction and Integrity Checks
=================================

Local verification
------------------

Create a local environment and run the Python tests:

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install -r bongard/requirements.txt
   .venv/bin/python -m pytest -q bongard/tests

Build the documentation and manuscript with warnings enabled:

.. code-block:: bash

   sphinx-build -W -b html docs docs/_build/html
   make -C bongard/manuscript

Lean is not a prerequisite.  Python is the canonical evaluator and replay
implementation.  An optional checker is non-authoritative and can be omitted.

Official corpus identity
------------------------

The active external paths are:

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

Pinned identities:

.. list-table::
   :header-rows: 1

   * - Object
     - SHA-256
   * - Archive
     - ``8c5542ac7b9ce8a6a14d157a0656dbde9da5b7843424eade4bd653759d9a27d0``
   * - Split file
     - ``ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230``
   * - Extracted corpus manifest
     - ``6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``
   * - Release descriptor
     - ``4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b``

Exposure discipline
-------------------

Do not run a live command as a smoke test.  Selection is consuming: the
successor exposure ledger is written, synced, and cold-read before any selected
PNG is opened.  A failed run remains exposed and is never rerolled.

At the pre-pilot ledger snapshot, 156 task IDs were exposed and 10,044
train/validation IDs remained exact-image-unseen.  The historical 10,047 count
belongs to the earlier post-Stage-A-A3 snapshot.  The completed 24-task pilot
leaves 180 exposed and 10,020 exact-image-unseen.  Strict reusable DRILL
capacity is zero.  Strict ``bd`` DEV capacity was 16 before the pilot and is 15
afterward because the selector protected task IDs but not shared semantic
disclosure keys.  That historical loss remains charged.  The prospective
selector excludes the complete disclosure-token closure and a metadata-only
regression preserved every baseline-viable DEV task without opening pixels.
Official test is sealed from model execution.

Coverage drill
--------------

``bongard.relational_coverage_drill`` is an offline, non-test engineering
command.  It selects only exact-unused train/validation tasks, persists
exposure before pixels, and records a selected-PNG manifest plus aggregate
loop/polygon/contact/obliqueness dispositions.  The completed pilot exposed a
guard bug: excluding exact DEV task IDs did not exclude tasks sharing their
semantic disclosure keys.  The prospective selector now enforces the complete
closure: family plus morphology for ``bd`` and pair plus attributes for
``hd``.  Its regression check was metadata-only and did not expose new pixels.

Its live invocation requires explicit write-once ledger and output stores:

.. code-block:: bash

   .venv/bin/python -m bongard.relational_coverage_drill \
       --corpus-root downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
       --split-file downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json \
       --corpus-manifest-digest sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138 \
       --ledger-in /absolute/path/to/current-ledger.exposure.json \
       --ledger-store /absolute/new/write-once/exposure-store \
       --output-store /absolute/new/write-once/report-store \
       --per-generator 1 \
       --per-split-family 4

This shows the completed pilot's selection parameters, not an instruction to
rerun its consumed cohort or reuse an existing store.  The corpus-manifest
argument is
``sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138``.
The output report digest is
``sha256:f78626c51b0af34cb0ccd96ed56041a51bcaeb453d3f26b10ea1ed1377542ae0``.
It records 336/336 successful panel extractions, 17,876 loops (10,354
substantive), 4,516 present versus 13,360 indeterminate polygon/obliqueness
observations, and 267,197 loop pairs with 46 contacts, 116,520 certified
separations, and 150,631 indeterminate contacts.

The selected-only closed-library replay is recorded compactly in
``bongard/data/relational_library_ablation_24task_outcome_v1.json``.  Its
record digest is
``sha256:ea6ee897513c22f1db8e656570e6572f2955855bbadb5caa39d8dc5dc8d423cd``
and it binds full-report output digest
``sha256:0a4b601ffc794a640175d2afda4f4b0d7f57fc980700bafbf09848ea4768c59b``.
The replay authenticated all 336 packet digests and evaluated the historical
2,520-query contact-inclusive diagnostic relational superlanguage in the fixed
forward orientation.  The result is 0/24
full seven-per-side separators, 0/168 exact 6+6 fold fits, zero held-out
generalizers, and a best task profile of 8/14.  It is explicitly
resubstitution/library coverage: official test, models, polarity flips,
negation rescue, and new exposure were all forbidden.

The corrected v2 closed-predicate library and support-only oracle are
implemented.  The authenticated A3 exercise froze all 65,678
proposer-reachable members and found exactly four support separators, all among
the 1,260 contact-disabled relational predicates; the 64,400 direct-count and
18 symmetry members contributed zero.  Every live plan must still run the oracle on its
authorized support packets and record whether no library separator exists,
whether the proposer was absent, or whether it missed or found an existing
separator.  This prevents a language miss from being scored as a model miss.
The initial union admits the existing direct topology/curvature counts,
relational queries, and bilateral symmetry thresholds as separate tagged
branches; it does not yet provide object-bound cross-branch conjunctions.

Cold replay contract
--------------------

There is no authorized production DEV plan.  The first metadata-only 15-task
plan was rejected before execution because its unkeyed support-index
commitment had only 49 possible preimages; brute force recovered both held-out
indices for every task.  It opened no DEV pixels and made no model calls.  A
replacement would require a fresh private schedule root and a hiding
commitment.  The hardened v4 runner and v4 campaign now implement and
fixture-test those controls together with official-manifest authentication,
immutable source pins, and crash-safe exclusive task claims.

The old 15-task DEV cohort must not be regenerated merely with better
commitments.  A metadata-only audit found 0/15 intended concepts fully
expressible in the current v3 two-closed-loop polygon/ratio language.  The
only currently qualified live target is the implemented explicit fixed
five-task exact-unused TRAIN representation-engineering mode with historically
exposed semantics.  Its admission and exposed-panel expressibility checks are
fixture-tested.  At this pre-run snapshot no real plan has been frozen or
published and no model campaign has executed.

A valid headless episode must retain enough canonical bytes to verify, without
a model:

* the exact selected task and predecessor/successor exposure chain;
* official-manifest membership, canonical path, and digest for every support
  and query PNG;
* immutable extractor, source, catalog, prompt, model/protocol, and calibration
  identities frozen as plan data;
* an exclusively locked durable task claim written before the one model call,
  with recovery treating an unfinished claim as terminal rather than retryable;
* complete typed packets and optional object-by-tag inventories;
* the one-shot positive predicate and full support evaluation;
* formula persistence before query release;
* prediction persistence before label reveal; and
* recomputed dispositions and final decision.

The four dispositions are ``present``, ``certified_absent``,
``indeterminate``, and ``error``.  Replay must never project the latter two to
Boolean false.

Historical runs
---------------

A1, A2, A3, and atomic attempts one through three are immutable consumed
records.  Their old command lines are audit evidence, not runnable recipes.
Do not replay them with the same seeds or paths.  A1 produced no scores, A2 was
invalidated by source drift, A3 failed its calibration-bin minimum, and the
atomic sequence stopped before any query score.

The post-hoc A3 support separator is reproducible: the base predicate is six
``present`` versus six ``certified_absent`` on the exact support mapping, and
the finite library has four exact support separators.  The intervening
5+1-versus-5+1 and zero-support-separator claim was a panel-mapping error.
Both held-outs are ``indeterminate``
under the 1/8 formulas; the 1/12 formulas certify the negative held-out absent
but leave the positive held-out
``indeterminate``.  The full seven-per-side library has zero exact separators.
Support resubstitution and held-out generalization must be reported separately.

The canonical mapping artifact is
``bongard/data/atomic_smoke_attempt3_relational_forensics_v1.json``, with record
digest
``0487edf805fda6de40ecfc42add1d8bf95e435e0f6912f6e2fd8d2a25e89eb2a``.
It binds all fourteen PNG hashes, the journal support order, support and
held-out source indices, every packet/base-result digest, and the four exact
support-formula digests.  Reproduction of any A3 count must start from that
mapping rather than reconstructing panel roles by filename position.

The complete-gate authority is
``bongard/data/a3_closed_language_gate_result_v2.json``, with record digest
``sha256:f9b6373df4dbe5d63807cf7e21be931db7ec0e9dfba106917df73d0e170a52d6``.
It records exactly four separators, all relational, and zero from the direct
and symmetry branches.  Python is canonical for this replay; Lean is optional
and removable.
