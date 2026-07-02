Thesis Overview
===============

Core Claim
----------

A fixed finite benchmark produces bounded adaptation. Open-ended evolution requires a generative ecology. Within that ecology, free energy can act as the local selection principle deciding which new structures persist.

The objective used throughout the repository is:

.. code-block:: text

   F_lambda(a) = R(a) + lambda C(a)

where ``R(a)`` is empirical risk and ``C(a)`` is a description-length proxy.

Why Fixed Benchmarks Are Insufficient
-------------------------------------

On a fixed finite benchmark, every program induces a finite behavior vector. Once evolution finds a compact solver for the best observable behavior vector, extra structure is punished by the complexity term without creating new payoff.

That means fixed-task free-energy minimization tends toward compression, not open-ended innovation.

Open-Ended Frame
----------------

The free-energy objective becomes open-ended only when risk is evaluated against an evolving frontier rather than a fixed list:

.. code-block:: text

   F_lambda,t(a) = exploitation_loss_t(a)
                 + alpha frontier_loss_t(a)
                 + beta interaction_loss_t(a)
                 + lambda C(a)

The environment distribution changes over time:

.. code-block:: text

   Q_{t+1} = G(Q_t, Population_t, Archive_t)

In short:

.. code-block:: text

   free energy selects locally;
   the ecology expands globally.

Developmental Overcapacity
--------------------------

The experiments also suggest that search should not be expected to enter the globally minimal basin directly. Some behaviors are found first through overbuilt machines, and only later can complexity pressure compress them.

This is not a failure of the complexity term. It is a basin-accessibility fact: complexity is costly, but the shortest program may be unreachable from cold stochastic search.

Lambda Sweeps
-------------

Following the loss-complexity structure-function viewpoint, ``lambda`` is swept instead of treated as a single tuned hyperparameter. The empirical object of interest is the loss-complexity frontier:

.. code-block:: text

   F_t(lambda) = inf_a [R_t(a) + lambda C(a)]

Open-endedness should appear as movement of this frontier across ecological time.

Abstraction Emergence
---------------------

The abstraction experiments test whether repeated structure can be encapsulated as a reusable predicate macro when doing so lowers free energy. The current result is deliberately narrow: it demonstrates encapsulation over a predefined primitive vocabulary, not discovery of the primitive observations themselves.

For the full source manuscript, see ``OPEN_ENDED_EVOLUTION_THESIS.md`` in the repository root.
