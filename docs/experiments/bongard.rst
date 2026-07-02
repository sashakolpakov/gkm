Bongard Direction
=================

The Bongard line has two roles in the project.

Internal Bongard-Style Rules
----------------------------

The local symbolic Bongard experiments use opaque-object sequences and sparse deterministic automata. They test whether rules such as equality at boundaries, duplicate detection, and length regularities can be discovered under clean train/validation/hidden/probe splits.

The important methodological result is that some tasks need developmental overcapacity: a larger search space can find a behavior that later selects down to a smaller exact solver, while a cold search capped at that final size fails.

Bongard-LOGO Adapter
--------------------

The external Bongard-LOGO adapter uses generated LOGO action programs without rendering images. It separates three modes:

* action skeleton features,
* derived macro features from action geometry,
* privileged metadata attributes.

Metadata mode is an upper bound, not a discovery result. The research target is to evolve or learn intermediate predicates, not to use metadata as the final solver.

Next Step
---------

The abstraction-emergence scaffold should be joined to Bongard-LOGO by evolving finite-state predicate recognizers over action-program observations. Those predicate recognizers should then be tested under the same controls used internally:

.. code-block:: text

   single-task
   unrelated-OR
   OR-factor
   no-share
   transfer
   oracle upper bound
