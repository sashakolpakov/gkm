GKM: Auditable Program-Growth Experiments
==========================================

This documentation summarizes the manuscripts, experiment notes, and executable
artifacts in the repository. Claims differ by subject: some domains perform explicit
lambda sweeps over finite model classes, while the ARC study reports a retained
program-growth history and replay-validated endpoints.

Subject directories
-------------------

Each subject has a canonical repository README:

* `ARC-AGI-3 <https://github.com/sashakolpakov/gkm/blob/master/arc/README.md>`_
* `Bongard <https://github.com/sashakolpakov/gkm/blob/master/bongard/README.md>`_
* `Colimit-cone core <https://github.com/sashakolpakov/gkm/blob/master/cone/README.md>`_
* `Foraging <https://github.com/sashakolpakov/gkm/blob/master/foraging/README.md>`_
* `Transduction <https://github.com/sashakolpakov/gkm/blob/master/transduction/README.md>`_

The central thesis is that open-ended artificial evolution requires more than a fixed benchmark. Free energy can act as the local selection rule, but the ecology must keep generating new validation pressures at the frontier of current competence.

.. important::

   The current abstraction experiments do **not** discover primitive perceptual atoms such as ``low_closure_error`` from raw input. Those atoms are predefined in the scaffold. The current result is narrower: a repeated conjunction over predefined primitive atoms is encapsulated as a reusable predicate macro when the free-energy accounting makes reuse cheaper than duplication.

.. toctree::
   :maxdepth: 2
   :caption: Manuscript

   thesis
   abstraction
   self_improving_agent

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   experiments/abstraction_emergence
   experiments/bongard
   experiments/reproduction

.. toctree::
   :maxdepth: 1
   :caption: Context

   related_work
