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

The Bongard track now targets the complete official image corpus. Its
canonical pipeline is panel pixels to provenance-bearing empirical witnesses,
then a closed positive predicate, then conditional verification. It records
four evidence dispositions, freezes a proposal before query release, commits
two predictions before label reveal, and cold-replays the result without a
model. Historical symbolic and small pilot scores are explicitly non-official.

.. important::

   Formal checking can certify the closed computation over recorded witnesses.
   It cannot certify that a vision model correctly interpreted the source
   pixels. Perceptual validity remains an empirical held-out question.

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
