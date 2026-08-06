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

The active track uses Python as the sole authoritative semantics for predicates,
the closed IR, evidence, calibration, synthesis, selection, evaluation,
persistence, replay, decisions, and every scientific result or artifact ID.
Lean is neither imported nor required.  Any checker consumes only an
already-frozen Python artifact and emits a detached, non-authoritative sidecar;
its presence, failure, disagreement, change, or deletion cannot alter Python
authority.  The
immediate problem is not macro syntax: it is obtaining useful typed direct and
soft observations from pixels, calibrating them honestly, and preventing
support-set artifacts or negation from masquerading as concepts.  See
:doc:`bongard` for the current design and status.
