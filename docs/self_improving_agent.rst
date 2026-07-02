Self-Improving Agent
====================

This page summarizes the self-improving agent line of the GKM program: an agent
that figures games out **on its own**, carrying a stock of human preconceptions,
inside a GKM / Goedel-machine loop. It is the culmination of the colimit-cone
view, where the reusable structures (the "legs") are no longer mined from a
fixed library but are **written by a proposer** and admitted by the same
free-energy accounting used elsewhere in the manuscript.

The Thesis
----------

The macro experiments isolate one half of the program: a free-energy rule that
admits a reusable abstraction only when it pays for itself, with candidate
structures supplied by enumeration. The complementary half is where the
candidate structures *come from* when they cannot be enumerated. Here the
reusable structures are whole programs of perception, mechanic discovery,
planning, and strategy, and they are produced by a proposer rather than picked
from a list.

The target environments are local ARC-style keyboard games, played only through
their runtime. The running example is the game ``wa30``.

The Rawest Substrate
--------------------

A recurring failure mode was smuggling the solution into the harness as a
"primitive." Any helper richer than the bare runtime interface (a perception
module, a typed object model, a push-versus-carry mechanic, a search planner) is
the programmer solving the game for the agent, and it does not transfer to a
different *type* of game.

The corrective is to expose only the boundary common to all games. The
``Arena`` substrate hands the proposer exactly four things:

.. code-block:: text

   step(action)        -> next frame
   frame               -> raw 64x64 integer grid
   levels_completed    -> scalar reward
   clone()             -> a copy for lookahead

Nothing in this interface mentions objects, colours, walls, avatars, or goals.
It is the only contract that survives a change of game type, so it is the only
contract the substrate provides. Everything else (perceiving objects, finding
the controllable object, learning the mechanic, inducing the goal, modelling an
autonomous helper, planning, composing manoeuvres) must be produced inside the
loop. None of it is hand-coded.

Human Preconceptions and the Proposer
-------------------------------------

The proposer is given a rich system prompt of **human preconceptions**: that
frames contain objects with persistent identity; that some objects are
containers and barriers partition space into regions; that one object may be a
controllable avatar and others may be autonomous agents with their own goals;
that cooperation can require a hand-off where one agent leaves something where
another can reach it; that reachability is constrained by barriers and must be
reasoned about per agent; that rewards may be sparse, so the agent should set
itself dense surrogate objectives and discover the affordances of its actions.

These priors are deliberately non-procedural. They describe what kinds of things
a world tends to contain, not how to solve any particular game.

Given those priors and the raw substrate, the proposer **writes its own code**:
a ``solve(env)`` program containing its perception, its mechanic probe, its
planner, and its strategy. The proposer is pluggable:

- a **small local model**, which is currently too weak to drive the loop; and
- the **Claude Code agent** invoked headlessly with file and shell tools and a
  tester, so it writes a solution, runs it on the real ``Arena``, reads the
  failures, and iterates -- the same write/run/fix loop a human programmer uses.

All reported results use the strong proposer.

Free-Energy Admission
---------------------

A written program is admitted only if it verifiably improves on the incumbent,
scored by the same free energy as the rest of the program:

.. code-block:: text

   F = R + lambda C

where ``R`` is driven by ``levels_completed`` on the real game and ``C`` is the
program's description length, with ``lambda`` small so that parsimony breaks ties
between equally capable programs without stifling a program that is large but
clears more levels. Compression-progress and disagreement signals supply an
intrinsic novelty term that *steers* exploration when the reward is flat, and
earlier admitted behavior is preserved against regression. The simulator is the
ground-truth verifier: a candidate is replayed on a fresh environment and kept
only if the reward actually advances.

The Source-of-Novelty Principle
-------------------------------

Two constraints govern the loop:

1. **Selection and pricing cannot be a source of novelty.** Free energy can only
   rank and compress structures it is given; the new structure must come from the
   proposer.
2. **Discovery and strategy must be learned inside the loop, never hand-coded.**

Together they place the burden of invention on the proposer and the burden of
honesty on the verifier. This is the stance of Schmidhuber's Goedel machine and
PowerPlay: a system that rewrites its own problem-solving code and adopts a
rewrite only against an empirical proof of improvement -- here discharged by the
simulator rather than by a formal theorem prover.

Result on ``wa30``
------------------

The probe-discovered context for ``wa30`` (the controllable avatar, the carrier
objects, the container region, the pick-up-and-carry mechanic, and the toggle
action) is established by interaction, not declared. Handed that context and the
preconception priors, the uncrippled Claude proposer wrote a single adaptive
``solve(env)`` program over roughly 4.2 hours of write/run/fix iteration (final
program 359 lines) that cracks ``wa30`` levels 1, 2, **and** 3. The result is
replay-validated: an independent replay of the recorded 288-move sequence on a
fresh environment reaches level 3, at ``F = -1.920``.

The proposer found the insights that had earlier been hand-coded in failed
attempts. An honest audit of *what was given*: the priors of these runs were not
fully neutral -- distilled from earlier human play of this game, they contained an
attach-carry-release experiment recipe and a relay-at-a-boundary hint, and the
interaction probe named the mechanic ("pick-up-and-carry") from a hand-coded verb
vocabulary. The priors have since been **neutralized** (generic world-priors only);
re-cracking ``wa30`` from scratch under them is the discriminating experiment.
Beyond the priors, the agent demonstrably found:

- It learned to **freeze** the container region at level start. A delivered
  carrier turns its slot the carrier colour and disappears from the interior
  detector; without freezing, delivered carriers are re-counted as loose and the
  goal signal is corrupted.
- On level 2 it learned to **complement** the autonomous helper by delivering the
  carriers farthest from the container, rather than competing for the same ones,
  then idling to let the helper tick forward.
- On level 3 it exploited the engine's **asymmetric** carry collision: a carried
  carrier can be placed onto a dividing-wall cell the avatar itself cannot enter,
  so the avatar relays each left-side carrier onto the wall column, where the
  right-side helper picks it up and ferries it to the container.

All three were discovered from raw frames plus the probed context plus the
priors, with no hand-coded leg. A later credited run extended the same solver
further, reaching ``wa30`` **level 6** (458 moves, replay-validated, ``F = -3.903``)
before a session limit stopped it. That run also exposed a limitation: asked in the
prompt to grow a reusable *leg library* (see below), the agent ignored the request
and instead grew a single monolithic solver -- the leg-library discipline has to be
enforced by the harness, not merely requested.

Honest Caveats
--------------

The result is bounded in two ways:

- The claim is precisely levels 1 through 6 of ``wa30`` and, for the same agent
  pointed at a **second** game (``ls20``, a slide-to-match mechanic), levels 1
  through 4 -- all replay-validated. It is not all levels and not all games:
  ``wa30`` levels 7-9 are unsolved. The ``ls20`` transfer is the intended evidence
  that the rawest substrate carries across game *types*.
- The loop currently needs a **strong** proposer. A system-prompt-only local
  model mis-reasoned two-sided reachability under barriers even with the priors
  spelled out, asserting feasibility where the interaction verifier proved boxes
  were stranded. The architecture (the rawest substrate, the preconception
  priors, the free-energy admission, and the simulator as verifier) is general
  and is the contribution; the open question is how weak a proposer that harness
  can lift to competence.

Growing a Leg Library
---------------------

Re-deriving every level from scratch is wasteful and, worse, misses the point. The
intended discipline is to grow a **library of legs** -- small, named, reusable
program fragments (perception helpers, navigation, an object-transport skill, a
relay-across-a-barrier skill) -- and to solve later levels by **composing** existing
legs with as little new structure as possible. On the early levels the agent still
invents legs, because it is learning the game's rules; by the later levels it should
recognise a level as an earlier one in a different geometric configuration that is
semantically the same, and clear it by combining known legs, adding almost nothing
new. The novelty then lives in the *combination*, and the agent iterates on the
composition far more than on the legs. After each level a **debrief** compares the new
solver to the previous ones, refactors repeated code into shared legs, and logs the
recurring composition pattern (itself a candidate higher-order leg).

This is exactly ``F = R + lambda C`` with the right complexity term: ``C`` is the
**marginal** novelty introduced -- the description length of *new* legs plus the
composition glue -- so a reused leg costs nothing and parsimony directly rewards
transfer. It is the same accounting as the macro experiments, now applied to
invented rather than enumerated structure.

Merely *requesting* this in the prompt was not enough (the agent grew a monolithic
solver instead), so the discipline is **enforced by the harness**: each level's player
is restricted to composition over a shared ``legs.py``, a separate refactor pass runs
after each level, and free energy is scored on marginal new-leg size. Run that way on
``ls20`` (capped at level 4), the enforced orchestration produced exactly the predicted
signature. Level 1 invents the library (marginal cost 55; a clone-BFS skill that
self-discovers which tiles to visit); levels 3 and 4 add essentially nothing (marginal
cost 2 each) and are literally one call to the level-advancing leg; level 2 reuses the
same leg with a single knob (a fuller state key, because its goal depends on carried
sprites), threaded in by the debrief so both players share one generic search. Cracking
ls20 levels 1-4 (replay-validated) cost total marginal novelty 77, front-loaded into
level 1 -- transfer, not re-derivation, and priced as such.

Relation to the Colimit-Cone Program
-------------------------------------

This substrate is the limiting case of the colimit-cone view that runs through
the repository. There, reusable navigation and interaction *legs* are discovered
by interaction rather than hand-coded, and a *cone* composes those legs into a
solution priced by reward against complexity. In the self-improving agent the
legs are no longer mined from a fixed motif library but are written by the
proposer as program fragments, the cone that glues them is the proposer's own
planner, and admission is still ``F = R + lambda C`` verified on the simulator.
The macro experiment and this agent experiment are two ends of one spectrum: the
same accounting decides whether a reusable structure earns its cost; what changes
is whether that structure is enumerated or invented.
