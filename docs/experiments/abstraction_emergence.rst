Historical Abstraction-Emergence Control
=========================================

This is a retained internal control, not a visual Bongard benchmark.  It does
not use the complete ``ShapeBongard_V2`` corpus and does not learn primitive
observations from pixels.

Question tested
---------------

The experiment asks whether conditional description-length accounting favors
one shared predicate macro when the same predefined conjunction is reused,
rather than paying for duplicated inline rule bodies:

.. code-block:: text

   objective = task loss + lambda * (library complexity + rule complexity)

The primitive Boolean atoms are supplied by the experiment.  Conditions
compare inline rules, a shared macro, macro syntax that repays the definition
at every use, and an oracle upper bound.  The historical result was that
repeated structure could make the shared macro cheaper, while single-use,
unrelated-disjunction, and no-sharing controls did not.

Claim boundary
--------------

This demonstrates an accounting incentive for encapsulation and reuse.  It
does not demonstrate visual grounding, semantic predicate discovery, Bongard
generalization, or self-improvement on the official corpus.  In particular,
it supplies the atoms that the active visual track must obtain from fallible
panel observations.

The original implementation and detailed reports are preserved at the
annotated Git tag ``pre-bongard-complete-rewrite-20260805``.  They should not
be combined numerically with the current Stage-A or Stage-B protocol.

Relationship to the active track
--------------------------------

The active track uses Python as the sole runtime semantics for predicates, the
closed IR, evidence, synthesis, selection, evaluation, persistence, replay, and
decisions.  Lean is neither imported nor required.  The strict ``un-Lean``
migration is nevertheless incomplete: old whole-tree provenance enters one
atomic lineage identity, a legacy config literal names Lean, and the unused
optional checker API is still in-process rather than isolated.  Those residues
do not affect the current Python execution, but must be removed before claiming
that every lineage ID is checker-byte-independent.

Atomic attempt three made the missing abstraction concrete.  Candidate-
independent prose retained useful object relations, but one surface-valid
``atom`` bundled shape, relative size, tilt, and directed attachment.  The
candidate-aware scorer then lost object roles on negative supports, so Python
found no exact separator and stopped before query access.  The immediate problem
is therefore not macro syntax: it is obtaining stable typed objects and
factorized relational micro-predicates from pixels, calibrating them honestly,
and preventing support-set artifacts or negation from masquerading as concepts.
See :doc:`bongard` for the current design and status.
