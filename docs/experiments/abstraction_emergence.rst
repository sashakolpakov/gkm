Historical Abstraction-Emergence Control
=========================================

Status
------

This experiment is a retained internal control. It is not a visual Bongard
benchmark, does not use the complete ``ShapeBongard_V2`` corpus, and does not
discover primitive observations from pixels.

Question tested
---------------

The scaffold asks a narrow question: when the same predefined conjunction is
used repeatedly, does conditional description-length accounting favor a
shared predicate macro over duplicated inline rule bodies?

.. code-block:: text

   F = task loss + lambda * (library complexity + task-rule complexity)

The candidate primitive atoms are supplied by the experiment. The discovery
claim, where supported, concerns encapsulation and reuse of those atoms rather
than perception or semantic grounding.

Controls
--------

.. list-table::
   :header-rows: 1

   * - Condition
     - Purpose
   * - ``inline``
     - Solve with primitive atoms and duplicated bodies
   * - ``shared``
     - Define a macro once and pay cheap calls on reuse
   * - ``no_share``
     - Permit macro syntax but repay its definition per use
   * - ``oracle``
     - Supply the target predicate directly as a privileged upper bound

The historical observation was that repeated shared structure could favor a
macro, while single-use, unrelated-OR, and no-share controls did not. This is
useful evidence that the accounting mechanism can express a reuse incentive.
It is not evidence that a visual system can find ``bird-like``, contact,
closure, or any other primitive from a panel.

Archived implementation
-----------------------

The original control, tests, and reports are preserved at the annotated Git
tag ``pre-bongard-complete-rewrite-20260805``. Stale working-tree copies remain
physically present pending explicit deletion, but they are excluded from the
canonical Bongard package and current reproduction commands. Checkout that tag
in a separate worktree to inspect or reproduce the historical mechanism; do not
mix its supplied-atom score with current visual benchmark results.

Relationship to the canonical track
-----------------------------------

The visual track replaces supplied atoms with provenance-bearing empirical
witnesses and typed registered legs. Reuse is credited only after the new leg
passes calibration, nuisance, near-miss, anti-memorization, and full archive
replay gates. See :doc:`bongard` for the current claim boundary.
