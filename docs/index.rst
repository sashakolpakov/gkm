GKM: Open-Ended Evolution Under Free Energy
================================================

This documentation is the readable Sphinx version of the manuscript and experiment notes in the repository.

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
