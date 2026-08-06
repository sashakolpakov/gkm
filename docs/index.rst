Gödel--Kolmogorov Machine
=========================

This repository studies verifier-gated program growth.  Each experiment has
its own claim boundary: deterministic replay can check a computation over
recorded observations, but it cannot prove that a vision model interpreted an
image correctly.

Current Bongard status
----------------------

The active Bongard track targets the complete official ``ShapeBongard_V2``
release.  Python alone is authoritative for predicate execution, calibration,
cold replay, benchmark decisions, and scientific artifact IDs.  Lean or any
other proof checker may only consume an already-frozen artifact as a detached
optional sidecar; installing or deleting it must not change any result,
decision, or ID.

The current experiment has a terminal first attempt and a distinct repaired
attempt:

* A1 failed before scoring.  All 48 proposer calls succeeded, producing 37
  accepted soft claims, 10 direct-only records, and one parser rejection, but
  all 37 scorer calls were transport errors.  There were zero scores, labels
  remained withheld, and no calibration, accuracy, or negation evidence was
  produced.  Its receipt is ``sha256:9aa247d9...40cb`` and terminal failure is
  ``sha256:a130d9e6...65b83``.
* A2 was a distinct repaired-protocol 48-task DRILL experiment, not an A1 retry.
  It used protocol ``sha256:2d9261c7...81ca``, fresh no-reroll seed
  ``eb031fe1...5a40b1``, and durable successor ``sha256:9b7cb7ee...5b1ce8``.
  A concurrent agent source edit after freeze invalidated the run.  It wrote no
  Stage-A terminal artifact; 48 proposer and 34 scorer launches were observed,
  but outputs were lost, labels were not revealed, and no semantic inference is
  valid.  Its incident file digest is ``sha256:4ace426b...7ccd4``.  The same
  cohort may not be rerun.
* Stage B is unauthorized by both A1 and A2.  Any separately authorized future
  run would remain descriptive.  The completed semantic audit leaves exactly
  24 BD + 0 constituent-disjoint HD = 24 DRILL units.  The earlier 28-unit
  upper bound did not project complete-A2 HD constituent exposures.  DEV
  against the full ledger has 16 BD + 0 HD units, so the default 24-task request fails before
  pixels and cannot authorize SEALED.
* New Stage-A receipts bind the complete executable Bongard Python source
  boundary.  Source drift
  after exposure now persists a typed failure with labels withheld.  Exact-ID
  caches reduced synthetic Stage-A replay from 161.15 s to 11.50 s and Stage B
  from 218.88 s to 51.10 s.
* Visual-semantic execution on the official SEALED/test split is hard-disabled.

No score from this sequence is an official benchmark result.  Old
symbolic, generated, and small-pilot results belong to an earlier protocol and
are preserved at the annotated Git tag
``pre-bongard-complete-rewrite-20260805``.

Subject directories
-------------------

Each subject has a canonical repository README:

* `ARC-AGI-3 <https://github.com/sashakolpakov/gkm/blob/master/arc/README.md>`_
* `Bongard <https://github.com/sashakolpakov/gkm/blob/master/bongard/README.md>`_
* `Colimit-cone core <https://github.com/sashakolpakov/gkm/blob/master/cone/README.md>`_
* `Foraging <https://github.com/sashakolpakov/gkm/blob/master/foraging/README.md>`_
* `Transduction <https://github.com/sashakolpakov/gkm/blob/master/transduction/README.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Manuscript

   thesis
   self_improving_agent

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   experiments/bongard
   experiments/abstraction_emergence
   experiments/reproduction

.. toctree::
   :maxdepth: 1
   :caption: Context

   related_work
