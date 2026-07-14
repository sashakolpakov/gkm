ARC-AGI-3 Program-Growth Audit
================================

This chapter documents the ARC-AGI-3 artifact study. A coding proposer writes
level solvers, a local simulator validates promoted behavior by replay, and the
repository retains the source and intermediate states needed to inspect growth.
The study concerns artifact provenance. It is not an official ARC-AGI-3
sample-efficiency or leaderboard evaluation.

Interface Scope
---------------

The local ``Arena`` supplies:

.. code-block:: text

   step(action)        -> next frame
   frame               -> raw 64x64 integer grid
   levels_completed    -> scalar reward
   clone()             -> independent state copy for lookahead

``clone()`` is a strong simulator oracle. Search routines can fork a state and
evaluate actions without spending steps on the retained trajectory. The official
ARC-AGI-3 environment wrapper documents ``reset()`` and ``step()`` but not arbitrary
state forking. Accordingly, the reported 596 and 393 actions are final replay-path
lengths. They exclude cloned exploration, failed trials, proposer calls, and compute.

The local interface does not label game objects or goals. That fact should not be
confused with prior-free discovery: prompts, discovered context, and development
history can still transmit mechanic-specific information.

Promotion Protocol
------------------

The retained solver has three principal parts:

* ``legs.py`` contains shared routines;
* ``players.py`` contains level entry points and composition glue; and
* ``solve.py`` dispatches to the appropriate player.

A proposal is executed, its action path is replayed from a fresh environment, and
only a validated state is promoted. Pre-debrief, recovered-path, interrupted, and
post-debrief workspaces are retained under ``wip_context``. This distinguishes a
successful literal plan from a later parameterized refactor.

Forbidden source and private-runtime inspection is blocked before execution. The exact
rejected payload is retained in ``blocked_attempts.log`` for audit, while the main
transcript records only the rejection. This ledger is excluded from execution-taint
checks because its commands did not run. The exclusion applies only to entries created
by that guard; historical WIP is not retroactively relabelled as blocked. Promoted files
from earlier runs are nevertheless scanned by the current checker and need no rebuild
when they remain clean and their replay still validates.

This guard responds to repeated observed misconduct. During stalled ``ft09`` work,
the Sonnet API proposer emitted two separate commands that accessed ``env._game`` and
enumerated the private runtime. Under the declared interface these are operational cheating
attempts because they seek evidence unavailable through frames and actions. The first
exact transcript is retained in the ``interrupted_a9a30e6e4da1`` level-1 WIP snapshot,
and the run is not promotion evidence. The repetition suggests that compliance can
deteriorate when observational progress stalls. Prompt instructions are consequently
not treated as enforcement; blocking and promotion-time taint checks apply to every
proposer model.

Source-Growth Statistic
-----------------------

For source text ``f``, let ``d(f)`` count nonblank, noncomment lines plus the number
of elements in Python list, tuple, set, and dictionary literals. The historical
checkpoint field ``marginal_C`` is

.. math::

   C_k = [d(legs_k)-d(legs_{k-1})]_+
       + [d(players_k)-d(players_{k-1})]_+.

This is positive **net retained-size growth per file**, not gross diff additions and
not semantic novelty. For example, replacing 100 charged units with 100 different
units in the same file contributes zero. Unchanged shared code also contributes zero.
Therefore a low value supports reuse only when source inspection identifies calls to
previously retained routines and fresh replay validates the composition.

The statistic is preserved because it is the one stored in the historical artifacts.
Changing it now would invalidate comparisons with those checkpoints. A future gross
diff or tree-edit ledger would need to be recomputed from paired snapshots and reported
as a different measure.

Promoted Endpoints
------------------

.. include:: generated/arc_artifacts.rst

The ``ls20`` checkpoint records ``43, 2, 45, 3, 72, 130, 67``. At L2 and L4,
the thin player entries call unchanged search routines, which makes the small net
growth values attributable to reuse in those two transitions. Larger entries retain
more source or literal plan information. The sequence is an auditable construction
history, not an estimator of Kolmogorov complexity.

The complete published ``wa30`` ledger is
``112, 78, 95, 47, 405, 225, 145, 204, 147`` for L1--L9. The manuscript audit
sidecar maps these entries to the clean early Git promotions, preserved later
promotion states, and final nine-level artifact. The root checkpoint is still the
unchanged operational resume state, so its record list contains only the transitions
retained after its resume base.

Prior and Source Audit
----------------------

The main ``wa30`` development history is not mechanic-blind. Earlier investigation
used the actual game source to diagnose the carry mechanism, and later prompts/context
contained carry and boundary-relay guidance derived from earlier human and agent work.
The fact that a later proposer did not directly open the source does not remove that
lineage contamination.

A later neutral-prior run independently reached ``wa30`` L1, and another proposer also
reached L1. These replications support rediscovery of the first-level mechanic under
different conditions. They do not validate neutral-prior discovery through L9. The
complete wa30 result is therefore a replay-validated solver artifact, not evidence of
blind nine-level mechanic induction.

Mathematical Scope
------------------

The manuscript places retained artifacts in a finite behavior-description plane and
uses compatible partial policies to formalize leg composition. This is a framework
for posing audits. The current ARC experiments do not compute a Kolmogorov structure
function, perform a complete lambda sweep, or establish a free-energy optimum. The
compatible-policy colimit is an elementary union theorem; empirical value must come
from concrete factorization and replay, not from the theorem alone.

Comparator Scope
----------------

OPINE-World publishes an executable transition model and cached level-entry states.
Conditioning a transition model on observed initial state does not disqualify it as a
world model. The defensible artifact-level distinction is narrower: OPINE does not
publish the same charged leg/glue reuse ledger, and its wa30 archive shows substantial
retained source growth. Complexity units and evaluation conditions differ, so this is
not a compute-matched ranking and does not show that OPINE is ``not a world model``.

Reproduction
------------

The promoted paths and artifact locations are documented in ``REPRODUCE_ARC.md``.
The replay procedure tests endpoint behavior. Reproducing the stochastic proposer
history, cloned exploration budget, and externally hosted model calls is a separate
experiment not supplied by the current artifact replay.
