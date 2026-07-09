Abstraction Emergence Experiment
=================================

Source files:

* ``bongard/run_abstraction_emergence.py``
* ``bongard/abstraction_emergence_report.md``
* ``bongard/test_abstraction_emergence.py``

Reproduce
---------

.. code-block:: bash

   python3 bongard/run_abstraction_emergence.py
   python3 bongard/run_abstraction_emergence.py --scenario multi --show-rules
   python3 bongard/run_abstraction_emergence.py --scenario or_factor --show-rules
   python3 -m unittest tests.test_abstraction_emergence

Conditions
----------

.. list-table::
   :header-rows: 1

   * - Condition
     - Meaning
   * - ``inline``
     - Solve each task directly using primitive atoms.
   * - ``shared``
     - Define a macro once and call it cheaply from rules.
   * - ``no_share``
     - Allow macro syntax, but charge the macro definition per use.
   * - ``oracle``
     - Supply the privileged task predicate directly; upper bound only.

Core Accounting
---------------

.. code-block:: text

   F = total task loss + lambda * (library complexity + task rule complexity)

The macro ``solid_loop`` has definition complexity:

.. code-block:: text

   1 macro overhead + 3 primitive atoms = 4.00

In the OR factoring task, the shared rule body costs:

.. code-block:: text

   1 rule overhead + 2 macro calls * 0.35 + 2 ordinary atoms = 3.70

Total shared complexity:

.. code-block:: text

   4.00 + 3.70 = 7.70

Whereas the inline DNF rule duplicates the primitive structure:

.. code-block:: text

   1 rule overhead + 4 atoms + 4 atoms = 9.00

Main Observations
-----------------

* Single-task pressure does not create a macro.
* ``has_curve OR thin`` does not create a macro.
* ``(solid_loop AND has_curve) OR (solid_loop AND thin)`` does create a macro.
* ``no_share`` kills the effect.
* Transfer uses the learned macro at lower marginal complexity.

This is the current strongest internal control for abstraction emergence in the repository.
