RoboArm: replay-gated tabletop manipulation
============================================

``rb01-v1`` is a standalone six-action tabletop manipulation experiment. It
uses the Gödel--Kolmogorov Machine pattern--proposal, bounded experiment,
retained-source revision, verification, and exact replay--without importing or
emulating an ARC runtime. The current result is a deterministic simulator
study; it is neither a physical-robot safety certificate nor evidence of
sim-to-real transfer.

Operational contract
--------------------

The round models the pinned Waveshare RoArm-M2-S transform chain, cylindrical
command-space inverse kinematics, swept robot and held-object collision, one
workpiece, a barrier, and a target bin. Bilateral jaw enclosure can attach the
workpiece; it can then be carried, released, settled by quasi-static gravity,
and scored by a sparse completion predicate.

.. list-table:: Six-action apparatus
   :header-rows: 1

   * - ID
     - Meaning
   * - 1
     - Decrease the selected coordinate
   * - 2
     - Increase the selected coordinate
   * - 3
     - Select the previous coordinate
   * - 4
     - Select the next coordinate
   * - 5
     - Open the gripper
   * - 6
     - Close the gripper

The selected coordinates cycle through azimuth, reach, and height. A rejected
command consumes its turn and atomically preserves the preceding legal state.

Observation and hardware boundary
---------------------------------

The operational sensor contract is ``rb01-roarm-c920-v3``. Each observation
contains an exact ``numpy.uint8`` RGB frame with shape ``(72, 128, 3)``. It is
a deterministic pinhole approximation of a separately connected Logitech
C920s source, with perspective, occlusion, articulated geometry, material
shading, shadows, and vignetting. Pixels contain no HUD, semantic palette,
object coordinates, telemetry, or success banner.

Telemetry is separate from the image. It distinguishes host T=104 command and
interlock state from stock-style T=1051 arm feedback: encoder angles,
firmware-derived XYZ, signed raw servo loads, torque-enable flags, and supply
voltage. Arm request/response and camera capture times are recorded separately
with their pairing skew. The packet does not invent metric jaw aperture,
contact force, collision category, attachment state, or goal state.

Proposal and safety boundary
----------------------------

Headless Codex is proposal-only. A generation can emit bounded declarative
``experiment`` or ``candidate`` scenarios, but it receives no connector,
environment handle, socket, clone handle, private mechanics source, canonical
route, browser implementation, or actuation authority. The trusted host owns
schema validation, isolated preflight, the deterministic safety finite-state
automaton, the one-use commit permit, stepwise camera/telemetry interlocks,
fresh-source verification, exact replay, and promotion.

Experiments never commit. A candidate becomes eligible only after an earlier
generation has recorded genuine operational failure and a later retained-source
revision proposes a complete safe goal-reaching sequence. The canonical
63-action mechanics route is test infrastructure, not discovery evidence.

Replay-validated result
-----------------------

Campaign ``rb01-roarm-c920-v3-zero-seed-20260731`` began from an empty retained
source lineage. Four live proposer generations authored seven scenarios and
the host executed 321 isolated preflight actions. The evidence includes an
empty grasp, a successful 45 mm pickup, a motion rejection, and rejected
collision-producing descents.

Generation 2 reached sparse completion as a non-committing experiment, but its
route retained three rejected descent commands. The safety automaton denied a
62-action candidate carrying those commands. Generation 4 removed them and
produced a 59-action candidate. The host committed it, verified the retained
source for 59 fresh actions, and independently replayed the same 59-action
acquisition boundary.

The checked-in viewer export records three failure replays and two successful
replays. Its manifest binds sensor schema 3, the campaign identity, source-tree
digest, promotion receipt, and lineage-profile receipt. The construction
profile reports positive net-growth complexity ``191, 323, 168, 172``,
conditional normalized-AST novelty ``3013, 4095, 4198, 2740`` bytes, and
``0, 1, 5, 8`` transitively invoked unchanged legs. This is a four-generation
construction profile for one promoted round, not a solved-level sawtooth.

Reproduction
------------

From ``roboarm/``:

.. code-block:: console

   python3 -m venv .venv
   .venv/bin/python -m pip install -e '.[test]'
   PYTHONPATH=src:. .venv/bin/pytest -q
   cd web
   npm ci
   npm test
   npm run build

The current suites contain 91 Python tests and 20 browser tests. The browser
is a replay viewer only: it cannot propose, preflight, actuate, repair, verify,
or promote a solver. Full campaign workspaces and local browser captures are
ignored; the portable versioned evidence is under ``web/public/campaign/``.

The `RoboArm README
<https://github.com/sashakolpakov/gkm/blob/master/roboarm/README.md>`_ links the
normative specification, operational and safety reports, hardware I/O
reference, commands, and current limitations.
