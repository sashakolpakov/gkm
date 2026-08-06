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

The scientific and operational record is explicit:

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
* A3 completed 22 proposer calls and 15 scorer calls, but its fixed upper bin
  had only six clusters against a minimum of eight.  It is a terminal
  underpopulated-bin failure, not a benchmark score.  Intended orientation was
  13/15 versus 2/15 for the complement; negation did not win.
* The first live atomic N=1 persisted exact-task exposure but neither a
  prediction nor a terminal.  Its successful-call count is irrecoverably
  unknown in 0--29.  The task is consumed without reroll, and the incident is
  operational rather than a Bongard result.
* Attempt two is frozen as a distinct successor.  Active predecessor
  ``sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a``
  is exactly one append after A3 ledger
  ``sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4``.
  The remaining universe is exactly nine IDs, digest
  ``sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed``.
  The pinned launcher stages before secrets and exposure; a fresh
  mode-``0700`` journal writes an intent before each of 29 transports, each
  result before the next, and its terminal before runner return.  Journals
  cannot be resumed or retried.  No live successor outcome is claimed.
* Stage B is unauthorized by A1, A2, and A3.  Strict DRILL capacity after A3 is
  zero; DEV remains 16 BD + 0 HD under the same ledger-disjoint policy.
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
