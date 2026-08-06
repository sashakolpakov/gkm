Gödel--Kolmogorov Machine
=========================

This repository studies verifier-gated program growth.  Each experiment has
its own claim boundary: deterministic replay can check a computation over
recorded observations, but it cannot prove that a vision model interpreted an
image correctly.

Current Bongard status
----------------------

The active Bongard track targets the complete official ``ShapeBongard_V2``
release.  Python is the sole authoritative semantics: it defines predicates,
the closed IR, evidence dispositions and projections, calibration, synthesis,
selection, evaluation, persistence, cold replay, decisions, and every
scientific result or artifact ID.  Lean is neither imported nor required.  Any
checker emits only a detached, non-authoritative sidecar; its presence, failure,
disagreement, change, or deletion cannot alter Python authority.

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
* Atomic attempt two ran once from commit ``d0864525...13c1``.  Its journal
  closed 13 intents/results: twelve neutral support descriptions and one valid
  proposal receipt.  All ten questions ended in the ``?`` demanded by the
  prompt, but the shared soft-cue parser rejected U+003F.  No support scoring,
  formula, query call, prediction, label reveal, or score occurred.  Run and
  terminal artifacts persisted and cold replay passed.  This is an
  implementation-contract failure, not vision or benchmark evidence.  Its
  exposure successor is
  ``sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d``.
* **Atomic attempt three is PRE-LIVE / PENDING.**  It binds that predecessor and
  an eight-ID universe, digest
  ``sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9``.
  Python is frozen around release authentication; a staged launcher performs a
  fixed non-Bongard preflight; all stores must be pristine; and an exclusive
  seed-independent canonical-path claim persists before secrets or exposure.
  No attempt-three outcome is claimed.
* Stage B is unauthorized by A1, A2, and A3.  Strict DRILL capacity after A3 is
  zero; DEV remains 16 BD + 0 HD under the same ledger-disjoint policy.
* Visual-semantic model execution on the official test split is hard-disabled.
  Complete-release authentication still hashes official-test bytes, but no
  test task or panel is selected, exposed to a proposer or scorer, evaluated,
  or scored.

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
