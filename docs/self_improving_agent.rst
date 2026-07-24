The Gödel--Kolmogorov Machine on ARC-AGI-3
===========================================

This chapter documents the Gödel--Kolmogorov Machine as a self-improving
solver-growth approach for ARC-AGI-3. In the Gödel--Kolmogorov Machine, a coding
proposer writes level solvers, a local simulator validates promoted behavior by
replay, and incumbent legs are tried before new code is requested. The name
Gödel--Kolmogorov Machine joins verifier-gated self-revision with
description-length selection; it does not assert proof-search optimality. The
abbreviation ``GKM`` is used below only after this full introduction. Artifact
provenance is the evidence contract for acquisition and reuse, not the method's
identity. The study is not an official ARC-AGI-3 sample-efficiency or leaderboard
evaluation.

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

One tagged scratch workspace admits only one orchestrator process at a time. An
OS-level lock rejects overlapping runs before artifact seeding, while checkpoint
recording upserts by level. Legacy repeated level rows are normalized on load or
save by retaining the last entry and subtracting superseded charges.

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

The expanded campaign retains 23 replay-valid endpoints. In addition to the two
complete games, ``ft09``, ``r11l``, and ``tr87`` reach L6; ``g50t`` reaches L5;
``ar25``, ``re86``, and ``sp80`` reach L4; ``cd82`` and ``m0r0`` reach L2; and
12 games reach L1. ``bp35`` and ``tn36`` have no promoted level. Every counted
endpoint uses the same promotion and WIP layout, but only ``wa30`` and ``ls20``
have complete manuscript sidecar histories.

The published `Competition-Mode scorecard
<https://arcprize.org/scorecards/9e166671-0953-42f3-89de-a0fd57d7b147>`_
scores **17.136507936507936%** over all 25 public games. Its distinct unweighted
raw coverage is **37/183 = 20.2186%**; its stored paths contain 1448 actions and
the scorecard used 1456 API actions after eight resets. Subsequent local promotions
raise artifact coverage to **67/183 = 36.6120%** and 2148 stored replay actions.
Those 30 clears replay independently but have not been folded into the earlier
public scorecard. Neither number measures clone-enabled discovery interaction or
proposer-compute cost.

The reset-window campaign began at 100% weekly allowance and froze paid solving in
the protected tail; no solver turn was launched below the predeclared 20% reserve.
Failures are charged to their reasoning-effort arm. Cold L1 acquisition cost 4 displayed points
for 3 medium clears and 18 points for 12 high clears. Retained-solver continuations
cost 39 points for 7 medium clears and 14 points for one high clear. The high
continuations were escalations of medium failures, so these are not matched cohorts.
They do show that high was highly effective for broad cold entry but did not establish
a general cost or compression advantage on hard continuations. The final bounded high
escalation cleared ``re86`` L4 for two displayed points. The resulting policy starts
fresh continuations on medium, retains clean WIP, and permits one high rescue after a
medium failure when live headroom remains. High's incremental rescue yield and total
charged points—not pooled effort averages—are the decision-relevant measurements.
Retrospectively, one of six qualifying high-after-medium turns rescued its target;
those six turns charged 12 displayed points. This is a small, adaptively selected
sample, but it favors a bounded fallback over high-by-default continuation.

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
retained after its resume base and totals 1243 under that narrower scope; the complete
publication history totals 1458.

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

The manuscript places retained artifacts in a finite behavior--description plane and
models source growth by inverse-shaped diagrams of typed interfaces, executable cells,
and attaching maps. Their colimit is a pushout. Monomorphic interface inclusions are
designated cofibrations, so cobase change preserves the incumbent source presentation;
an optional debrief is treated separately as a replay-equivalent refactor. A finite
replay semantics supplies the empirical comparison map rather than assuming that
execution preserves arbitrary source colimits globally.

The compute-completeness result is conditional on deterministic finite games, finite
winning traces, recognizable replay, fair dovetailed search, or the stated stagewise
full-support proposer bound. It is an existence result, not a practical waiting-time
estimate. The current ARC experiments sample a computable complexity coordinate but do
not estimate the full Kolmogorov structure function, perform a complete lambda sweep,
or establish a free-energy optimum.

Comparator Scope
----------------

The common unit is the retained state that actually cleared a level. Interim
synthesis revisions, repeated same-level commits, and within-level notebook edits
are excluded. The resulting comparison separates cumulative executable size,
conditional novelty, operational reuse, and descriptive memory.

The cross-system marginal is the zlib-9 length of normalized top-level AST
statements in the current winning program that are not literal members of the
preceding winning program. A half-or-more decrease is a sharp drop. It is
attributed to reuse only when the winning entry point directly calls a named
definition whose normalized AST is unchanged from the preceding winning
checkpoint.

.. list-table:: Solved-checkpoint-only evidence
   :header-rows: 1
   :widths: 14 23 25 38

   * - System
     - Boundary coverage
     - Conditional AST marginal
     - Direct literal reuse
   * - GKM
     - 63 exact winning sources across 67 clears; 39 exact adjacent transitions
     - 21 of 39 comparable marginals decrease; 6 fall by at least half
     - Fourteen winning players directly call unchanged leg literals. Four
       transitions couple such a call to a sharp drop: ``ar25`` L2, ``g50t`` L4,
       ``ls20`` L7, and ``m0r0`` L2.
   * - OPINE-World
     - 146 pre-solve engines for 153 positive-reward trace events; 121 adjacent
       transitions
     - 49 of 115 comparable marginals decrease; 14 fall by at least half
     - Four synthesized-planner wins directly call unchanged engine literals.
       ``lp85`` L4 and ``tu93`` L3 couple such calls to sharp drops.
   * - baseline1 GPT-5.5 xHigh
     - 160 post-solve snapshots for 174 clears; 50 exact winning sources and
       18 exact adjacent transitions
     - 5 of 8 comparable marginals decrease; none falls by half
     - Zero. All 18 exact adjacent winning commands are fresh literal action
       programs and invoke no retained world-model definition.
   * - Retrodict
     - 170 solved memory checkpoints
     - No executable marginal is released
     - Zero executable witnesses. The released checkpoints contain curated
       playbook memory and limited scratch Python, not winning entry points.

The exact winning-entry-point test is asymmetric: OPINE has hard
level-to-level executable reuse; baseline1 does not, and Retrodict does not
release an executable winning entry point on which to run the test. GKM has the
strongest literal-leg evidence in the measured exact set.

The blanket claim that OPINE solves every level anew is false: ``lp85`` L4 uses
the identical winning planner literal from L3 and directly calls three unchanged
engine definitions while its conditional marginal falls from 5818 to 2550.
``tu93`` L3 is a second sharp-drop/reuse witness. baseline1's four exact
cumulative authored-source contractions remain real artifact contractions, but
they are not reuse witnesses under the winning-entry-point rule. Retrodict
supports memory transfer only. GKM has the largest number of direct literal-leg
wins in the measured exact set and integrates reuse-first execution with
marginal description accounting.

One non-sharp direct witness is ``ft09`` L6: its winning player calls the unchanged
``solve_coupled_key_board`` acquired at L5. The conditional AST marginal falls
from 5008 to 3730 bytes, which is a decrease but not a half-or-more drop. Its
historical two-unit ``marginal_C`` charge is a different source-growth statistic.

Machine-readable results are retained in ``arc/audit_results/``. The coupled
test is ``audit_marginal_literal_reuse.py``. The boundary analyzers are
``audit_gkm_solved_checkpoints.py``,
``audit_baseline1_artifacts.py``, ``audit_opine_solved_checkpoints.py``, and
``audit_retrodict_artifacts.py``.

Reproduction
------------

The manuscript and figure sources are documented in ``arc/manuscript/README.md`` and
build with ``make -C arc/manuscript``. The promoted paths and artifact locations are
documented in ``REPRODUCE_ARC.md``.
The replay procedure tests endpoint behavior. Reproducing the stochastic proposer
history, cloned exploration budget, and externally hosted model calls is a separate
experiment not supplied by the current artifact replay.
