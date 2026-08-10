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
state forking. Accordingly, the reported 597 and 365 actions are final replay-path
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

The prospective campaign runner also supports a bounded failure-revision treatment.
Each revision round runs in a fresh ephemeral proposer thread and is accepted only
when its protocol digest, frontier binding, round order, sealed diagnostics, and
terminal or promotion evidence authenticate as one aggregate. Unknown control fields,
cross-frontier evidence, incomplete diagnostics, and exhausted or tainted aggregates
fail closed. This is current campaign infrastructure, not a retroactive property of
the frozen 181-boundary release: that release remains governed by the verifier and
control revision named in its schema-v2 receipt.

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

The frozen v2 `Competition-Mode scorecard
<https://arcprize.org/scorecards/cf75e14b-2c25-41cb-bc70-53bd57411edb>`_
scores **98.11664037825032%** over all 25 public games. Its distinct unweighted
raw coverage is **181/183 = 98.907103825137%**; its certified paths contain 7001
actions and the scorecard used 7069 API actions including resets. The preceding
`ONLINE shakedown
<https://arcprize.org/scorecards/e293eeae-c0de-4263-a916-0a40ad282cbc>`_
validated the same frozen endpoint set. Neither score measures clone-enabled
discovery interaction or proposer-compute cost.

GKM is universal at the producer level: every game uses the same proposer contract,
Arena interface, blank scaffold, complexity coordinate, and replay gate. The learned
programs retained by that producer are game- and level-dependent executable outputs.

Marginal complexity by game and level
-------------------------------------

.. include:: generated/marginal_complexity_by_level.rst

The uniform ``ls20`` ledger is ``40, 54, 86, 114, 138, 170, 158``. Its sharp
conditional-AST drop at L2, from 737 to 247 compressed novelty bytes, is coupled to a
direct call of the unchanged ``follow_cardinal_runs`` leg. The uniform ``wa30``
ledger is ``43, 20, 32, 50, 39, 23, 28, 34, 49``. These are auditable construction
histories, not estimators of Kolmogorov complexity.

Prior and Source Audit
----------------------

The frozen release has 181 replay-verified endpoint wins. The stricter source audit
admits 174 exact winning-source checkpoints; it excludes ``ft09`` L2 and ``tr87``
L1--L6 deterministic reconstructions from source-marginal and reuse counts.

The frozen ``wa30`` history is a fresh, uniform L1--L9 reacquisition with an exact
winning-source boundary and replay-validated promotion manifest at every level. An
earlier exploratory lineage included mechanic-specific human and source-derived
context; it remains explicitly superseded provenance and contributes no boundary or
complexity increment to the uniform history. The release-wide taint and manifest gates
apply the same exclusion rule to every game.

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

The numerical comparison is generated rather than duplicated here. See
``arc/manuscript/generated/comparator_stats.md`` for the current table and
``arc/manuscript/opine_world_comparison.md`` for scope, caveats, and interpretation.
Its reproduction entry point is ``arc/manuscript/scripts/reproduce_manuscript.py``;
the underlying machine-readable audits are retained under ``arc/audit_results/``.

Reproduction
------------

The manuscript and figure sources are documented in ``arc/manuscript/README.md`` and
build with ``make -C arc/manuscript``. The promoted paths and artifact locations are
documented in ``REPRODUCE_ARC.md``.
The replay procedure tests endpoint behavior. Reproducing the stochastic proposer
history, cloned exploration budget, and externally hosted model calls is a separate
experiment not supplied by the current artifact replay.
