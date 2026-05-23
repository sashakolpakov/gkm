Abstraction Emergence
=====================

What Is Predefined?
-------------------

In the current scaffold, primitive observations are predefined. Examples include:

.. code-block:: text

   low_closure_error
   high_hull_fill
   turn_balanced
   has_curve
   thin

These are hand-defined boolean atoms. The experiment does **not** discover them from pixels, turtle traces, or raw geometry.

What Is Discovered?
-------------------

The discovered or selected structure is a reusable macro predicate over those primitive observations:

.. code-block:: text

   solid_loop = low_closure_error AND high_hull_fill AND turn_balanced

The system can solve tasks either by duplicating the full primitive conjunction inline, or by defining ``solid_loop`` once and calling it cheaply.

Why Simple OR Is Not Enough
---------------------------

A simple disjunction does not create abstraction pressure:

.. code-block:: text

   has_curve OR thin

There is no repeated substructure. The best rule is simply:

.. code-block:: text

   (has_curve) OR (thin)

The shared-library condition therefore selects no macro.

Why ``(A AND B) OR (A AND C)`` Matters
--------------------------------------

A factoring task does create abstraction pressure:

.. code-block:: text

   (A AND B) OR (A AND C)

Here ``A`` is repeated. In the experiment:

.. code-block:: text

   A = solid_loop
   B = has_curve
   C = thin

Expanded inline, the task is:

.. code-block:: text

   (low_closure_error AND high_hull_fill AND turn_balanced AND has_curve)
   OR
   (low_closure_error AND high_hull_fill AND turn_balanced AND thin)

The selected macro version is:

.. code-block:: text

   (solid_loop AND has_curve) OR (solid_loop AND thin)

Key Result
----------

.. code-block:: text

   or_control/shared: no macro, complexity 3.00
   or_factor/inline: no macro, complexity 9.00
   or_factor/shared: solid_loop macro selected, complexity 7.70
   or_factor/no_share: no macro, complexity 9.00
   or_factor_transfer/shared: transfer complexity 2.35 vs inline 5.00

Interpretation
--------------

The result is not that macro syntax itself helps. The ``no_share`` ablation allows macro syntax but charges the full macro definition at each use; under that accounting, the macro disappears.

The result is specifically that shared description length can make encapsulation favorable under the free-energy objective.

Current Limit
-------------

The experiment demonstrates predicate encapsulation and reuse. It does not yet demonstrate evolution of a predicate recognizer from raw input. The next step is to evolve finite-state predicate automata over action-program observations and run the same single-task, unrelated-OR, OR-factor, no-share, and transfer controls.
