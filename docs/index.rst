Gödel--Kolmogorov Machine: Verifier-Gated Program Growth
========================================================

This documentation describes experiments in verifier-gated program revision,
description-length selection, and cumulative library growth. Claims are scoped
per subject: a replay-validated internal mechanism is not automatically an
external benchmark result, and a formal symbolic derivation is not a proof of
perceptual correctness.

Subject directories
-------------------

Each subject has a canonical repository README:

* `ARC-AGI-3 <https://github.com/sashakolpakov/gkm/blob/master/arc/README.md>`_
* `Bongard <https://github.com/sashakolpakov/gkm/blob/master/bongard/README.md>`_
* `Colimit-cone core <https://github.com/sashakolpakov/gkm/blob/master/cone/README.md>`_
* `Foraging <https://github.com/sashakolpakov/gkm/blob/master/foraging/README.md>`_
* `Transduction <https://github.com/sashakolpakov/gkm/blob/master/transduction/README.md>`_

The Bongard track now targets the complete official image corpus. Pure Python
defines the canonical predicate, evaluation, and replay semantics; Lean is an
optional, removable cross-check rather than part of the benchmark's semantic
boundary. The protocol freezes a proposal before query release, commits two
predictions before label reveal, and supports deterministic replay without a
model. Historical symbolic and small pilot scores are explicitly non-official
and are preserved at the annotated Git tag
``pre-bongard-complete-rewrite-20260805``.

The current PURE support-prototype baseline failed development calibration.
Its interval boxes discard correlation between preprocessing scenarios, one
centroid per side does not represent multimodal near-miss classes, the
one-group proposal restriction prevents cross-group concepts, and the neutral
raster features do not provide the semantic and relational vision needed by
many panels. This negative result diagnoses the representation; it is not a
complete-corpus benchmark score.

.. important::

   Mechanical checking can certify the closed computation over recorded
   witnesses. It cannot certify that a vision model correctly interpreted the
   source pixels. Perceptual validity remains an empirical held-out question,
   whether or not the optional Lean cross-check is enabled.

.. toctree::
   :maxdepth: 2
   :caption: Manuscript

   thesis
   abstraction
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
