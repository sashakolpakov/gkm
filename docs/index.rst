Gödel--Kolmogorov Machine
=========================

This repository studies verifier-gated program growth.  Deterministic replay
can verify a computation over recorded observations; it cannot prove that a
fallible observer interpreted an image correctly.  Each experiment therefore
states its empirical boundary and its claim boundary separately.

Current Bongard track
---------------------

The target design for any independently authorized future cohort is
Python-first and fail-closed:

.. code-block:: text

   pixels -> typed geometry plus calibrated soft observations
          -> deterministic support-consistent version space
          -> typed semantic gap if empty
          -> rank verified candidates if nonempty
          -> frozen Python predicate -> query evaluation -> model-free replay

The latest skeleton-graph engineering line established a fixed-32 development
observer, an exact passed-fit authority, role-free raw-inference custody, and a
staged delayed-label calibration runner.  The calibration campaign itself did
not launch.  An overbroad repository search exposed the complete 4,400-task HD
action-program authority before the prediction barrier, so the incident was
persisted as a write-once tombstone and the campaign ended in a typed
**custody GAP**.

That GAP is not evidence for an empty semantic version space: no support matrix
or inventory was constructed, its version-space counts are null, and pixel,
model, label, ranking, and query activity stayed closed after terminalization.
Official-test images and labels remain unopened, but the HD program-semantic
boundary is no longer sealed and must not be described as such.

The earlier loop, soft, and five-task campaigns remain historical engineering
evidence.  Their obsolete live soft/prompt implementations have been removed
and preserved only as authenticated inert source preimages.  See the
:doc:`current Bongard authority and history <experiments/bongard>`.

Current RoboArm track
---------------------

The standalone ``rb01-v1`` experiment applies replay-gated program growth to a
deterministic tabletop manipulation round. A proposal-only Codex process emits
bounded scenarios; the trusted host owns preflight, safety authorization,
actuation, fresh-source verification, and exact replay. The promoted v3
campaign uses separate 128x72 RGB camera bytes and stock-style RoArm feedback,
with no ARC runtime dependency and no sim-to-real claim. See :doc:`the RoboArm
experiment <experiments/roboarm>`.

Subject directories
-------------------

Each subject has a canonical repository README:

* `ARC-AGI-3 <https://github.com/sashakolpakov/gkm/blob/master/arc/README.md>`_
* `Bongard <https://github.com/sashakolpakov/gkm/blob/master/bongard/README.md>`_
* `Colimit-cone core <https://github.com/sashakolpakov/gkm/blob/master/cone/README.md>`_
* `Foraging <https://github.com/sashakolpakov/gkm/blob/master/foraging/README.md>`_
* `RoboArm <https://github.com/sashakolpakov/gkm/blob/master/roboarm/README.md>`_
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
   experiments/roboarm

.. toctree::
   :maxdepth: 1
   :caption: Context

   related_work
