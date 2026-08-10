# Godel-Kolmogorov machine–RoboArm: Implementation Specification

**Working game ID:** `rb01-v1`
**Repository scope:** `sashakolpakov/gkm/roboarm`
**Primary target:** a self-contained Python environment implementing an ARC-style observation/action/reward contract for a genuine headless-Codex Godel-Kolmogorov machine campaign, plus a downstream browser viewer for failed and successful attempt replays
**Status:** historical design and acceptance specification

> **Implementation note.** This document preserves the full v3 design envelope,
> including planned multi-level, manual-play, oracle, and future-physics work.
> The checked-in release implements the single operational `pick-place-v2`
> round and the proposal/FSA/replay path described in `README.md`,
> `OPERATIONAL_SLICE_REPORT.md`, and `SAFETY_FSA_REPORT.md`. Those documents and
> executable tests are authoritative for current behavior where this original
> specification describes a broader or superseded target.
**Physical reference:** Waveshare RoArm-M2-S, Reichelt article WS-25974

---

## 0. Executive decision

Build `roboarm` as a **real local ARC-AGI-3-style game**, not as a separate Gym environment and not as a disposable toy simulator.

### 0.1 Repository isolation — normative priority

Every new or modified file required by this project must live below:

```text
/Users/sasha/gkm/roboarm/
```

This includes simulator source, environment facades, launchers, prompts,
source-access guards, tests, caches, campaign workspaces, replays, reports, and
generated artifacts.

The parent repository and all sibling directories, including `arc/`, are
**read-only research references**. They may be inspected to understand the
Godel-Kolmogorov machine experimental discipline, but this project must not add, modify, delete, copy
generated artifacts into, or depend on writes to them.

If any later requirement conflicts with this isolation rule, this rule wins.
Implement the required behavior in `roboarm` itself. Do not solve an integration
problem by patching files outside `roboarm/`.

### 0.2 ARC-style, not ARC-API-bound — normative priority

`ARC-style` means only the public experimental contract:

- an exact 128×72×3 RGB8 camera frame plus synchronized structured controller
  telemetry;
- a small discrete action vocabulary;
- turn-based `reset()` and `step()`;
- sparse `levels_completed` reward;
- deterministic replay;
- exact simulation cloning; and
- no privileged world-state channel.

The implementation must not import, install, wrap, subclass, or depend on
`arc_agi`, `arcengine`, `ARCBaseGame`, an ARC game loader, ARC metadata,
ARC sprites, or ARC action classes. It owns its Python protocol directly.

However, do **not** make the robot embodiment itself the principal concealed puzzle. The default experiment is a **known-actuator / unknown-scene** experiment:

- the agent is explicitly given the action meanings;
- the active control axis and commanded/measured gripper pose are consistently
  reported in public controller telemetry;
- the arm controller is deterministic and documented;
- the simulator internally uses the published RoArm-M2-S joint chain and fitted collision geometry;
- scored difficulty begins with how objects and scenes respond to commanded gripper motion.

The research target is the transducer of the manipulated world:

```text
known gripper command
    + current visual/contact context
    -> object motion, collision, enclosure, attachment, slip, fall, occlusion,
       support, placement, and resulting scene
```

It is **not** primarily:

```text
unknown button
    -> unknown robot-joint motion
```

The game must still expose the normal ARC-style external contract:

- a 128×72×3 `uint8` RGB camera observation;
- a separate public controller-telemetry packet synchronized to the frame;
- a small discrete action set;
- turn-based `reset()` / `step()` interaction;
- `levels_completed` as the external reward;
- deterministic replay;
- exact state cloning in simulation;
- no privileged object coordinates, grasp flags, contact identities, or hidden physics state.

Internally, implement a deterministic **2.5D tabletop manipulation simulator** with:

- a RoArm-M2-S articulated chain calibrated from Waveshare's official ROS2/Xacro model;
- a documented end-effector controller mapped to the articulated chain;
- joint limits and rejected unsafe commands;
- link/table/object collision;
- pushing and obstruction;
- bilateral grasp detection rather than proximity-based “magic pickup”;
- rigid attachment after valid grasp;
- gravity, support, release, and optional slip;
- obstacles, occlusion, object geometry, load/contact telemetry, and seeded variation;
- a calibrated fixed perspective camera producing the RGB observation.

The release must also make genuine campaign behavior inspectable in a browser.
The browser is downstream of the Godel-Kolmogorov machine: it loads saved unsuccessful and successful
attempt evidence and illustrates what the proposer-generated solver actually
did. It is not a controller, planner, discovery loop, or alternate success
path. The host-owned Python RoboArm connector remains authoritative for every
observation, preflight, action transition, sparse reward, and replay used by
the machine. The proposer never receives that connector. The viewer shows the
exact 128×72 RGB camera input, its exact companion telemetry packet, and a
larger state-synchronized 3D reconstruction of the same attempt. The browser
reconstruction is explanatory replay output, never scientific input.

The standalone environment facade should remain thin. Most engineering effort
belongs in the `roboarm` world and its object/scene mechanics, not in experiment
glue and not in hiding the robot controls.

### 0.3 Proposal-only LLM and deterministic safety FSA — normative priority

The coding LLM is an untrusted proposer, not an actuator and not a verifier. It
must receive no live environment object, connector client, Unix socket, token,
clone handle, physical adapter, or direct `step()` capability. Headless Codex
may write retained proposal code and bounded declarative scenarios containing
only:

- a scenario identifier;
- role: `experiment` or `candidate`;
- a falsifiable hypothesis;
- an expected public observation; and
- a finite list of legal public action IDs.

It may not author an observed frame, reward, terminal state, `passed`,
authorization, safety decision, or verified outcome. Those fields are
host-owned.

Every scenario passes through a deterministic finite-state safety automaton:

```text
proposal
  -> closed-schema validation
  -> isolated authoritative digital-twin preflight
  -> deterministic trajectory/load/collision/budget verification
  -> optional single-use in-memory commit permit
  -> stepwise interlocked connector execution
  -> observed-fact sealing
  -> fresh source and exact-path replay
```

`experiment` scenarios are preflight-only and can never commit. `candidate`
scenarios may commit only after a complete safe goal-reaching preflight and an
earlier generation's genuine failed observation. Before each committed action,
the connector must reproduce the next transition on a clone and compare it to
the admitted preflight. A simulator-rejected collision attempt is genuine
negative evidence, but the known-rejected command must not enter the committed
trace.

The FSA must not encode the puzzle solution, target coordinates, grasp recipe,
or canonical action path. It enforces generic safety, provenance, resource,
state-transition, and replay invariants only. This proposal/observation/verdict
separation follows the deterministic connector pattern studied in
`sashakolpakov/bayesilisk`; it is implemented locally and introduces no
Bayesilisk runtime dependency.

## 1. Scientific purpose

The central experiment is:

> Given a known and stable robot-control interface, can the existing verifier-driven Godel-Kolmogorov machine program-growth loop acquire compact, reusable transducers for how objects and scenes respond to contact and manipulation?

The mechanisms of interest are:

- free-space approach versus contact;
- direction-dependent pushing;
- friction and obstruction;
- jaw enclosure and grasp preconditions;
- attachment and object–gripper co-motion;
- lift clearance;
- collision while carrying;
- release, falling, support, and settling;
- occlusion and persistence;
- transfer across object position, dimensions, shape, friction, and clutter.

The robot should function as a **known experimental instrument**. Its internal articulated geometry remains physically meaningful so that commands can be rejected by reachability and collision, but the agent is not scored on rediscovering the action map or deriving inverse kinematics.

The intended progression is:

1. Build and validate the known actuator/controller and articulated simulator.
2. Treat arm-only motion tasks as calibration and regression tests.
3. Start the scored curriculum with object interaction.
4. Run the current Godel-Kolmogorov machine loop with only a small interface appendix.
5. Measure acquisition, replay validity, mechanism reuse, and held-out scene generalization.
6. Add controlled realism and partial observability.
7. Only then build a physical RoArm adapter exposing the same known control contract.

The first simulator result is **not** a sim-to-real claim. It is evidence that the method can acquire world-response mechanisms in a structured embodied substrate.


## 2A. Exact physical reference: Waveshare RoArm-M2-S

Use the specific arm sold by Reichelt as article `WS-25974`, not a generic four-axis arm. The coding agent must treat the following as the hardware reference contract.

### 2A.1 Manufacturer-level envelope

Use these published product values for configuration, sanity checks, and realism modes:

```text
model                         Waveshare RoArm-M2-S
nominal architecture          base yaw + shoulder pitch + elbow pitch + gripper
advertised DOF                4
physical servo count          5 (dual-drive shoulder plus base, elbow, gripper)
base travel                   360 degrees
shoulder travel               approximately 180 degrees
elbow travel                  approximately 180 degrees
horizontal workspace          up to 1.090 m diameter
vertical workspace            up to 0.798 m
rated payload                 0.5 kg at 0.5 m
repeat positioning accuracy   approximately +/- 4 mm under the same load
no-load servo speed           40 rpm
servo encoder                 12-bit magnetic encoder over 360 degrees
servo torque                  30 kg.cm at 12 V per listed servo
arm mass                      0.826 +/- 0.015 kg, excluding table clamp
power                         12 V, 5 A
```

Do not force every advertised value into canonical gameplay. Use them to keep generated scenes physically plausible and to define optional load, speed, and endpoint-error regimes.

### 2A.2 Official URDF/Xacro reference

The primary kinematic source is Waveshare's official `roarm_ws` repository:

```text
src/roarm_main/roarm_description/urdf/roarm_m2/roarm_m2.xacro
```

Import or transcribe the following transforms and limits into a versioned hardware profile. Values are in metres and radians.

```python
ROARM_M2_PROFILE = {
    "base_to_yaw_origin_xyz": (0.0100000008759151, 0.0, 0.123059270461044),
    "shoulder_to_elbow_origin_xyz": (0.236815132922094, 0.0300023995170449, 0.0),
    "elbow_to_gripper_origin_xyz": (0.002906, -0.21599, -0.00066683),
    "elbow_to_tcp_origin_xyz": (0.002, -0.2802, 0.0),
    "base_yaw_limits": (-3.1416, 3.1416),
    "shoulder_limits": (-1.5708, 1.5708),
    "elbow_limits": (-1.0, 2.95),
    "gripper_limits": (0.0, 1.5),
}
```

The corresponding fixed frame rotations in the Xacro must be respected when implementing exact forward kinematics. Do not treat the XYZ offsets above as if they were all expressed in one unrotated planar frame.

For the fast simulator, it is acceptable to precompute a reduced analytic model from those transforms, but it must be validated numerically against a direct homogeneous-transform implementation at randomly sampled legal joint states.

### 2A.3 Fidelity hierarchy

Use this order of authority:

1. Official Waveshare RoArm-M2 Xacro for joint origins, frame rotations, TCP, and software joint limits.
2. Reichelt/Waveshare product specifications for workspace, payload, speed, encoder resolution, repeatability, and total assembled mass.
3. Simplified fitted collision primitives for fast deterministic scene mechanics.
4. Tuned object-contact coefficients for useful benchmark behavior.

Do not use product photographs to guess dimensions already present in the Xacro. Do not use the full STL meshes in the canonical simulation loop; they are unnecessary and would reduce speed and portability.

### 2A.4 Scientific boundary

Hardware fidelity is used to constrain which contacts and motions are physically possible. It must not turn arm calibration into the task. The agent is still given the command-space semantics and does not need to infer servo IDs, frame conventions, inverse kinematics, or serial protocol details.

## 2B. Non-goals

Do not turn v1 into a general robotics platform.

Out of scope for the first complete version:

- ROS, MoveIt, Gazebo, Isaac Sim, PyBullet, or MuJoCo as hard dependencies;
- neural perception, pretrained object detectors, or imitation learning;
- continuous-control reinforcement learning;
- exact motor electronics or servo thermal simulation;
- photorealistic rendering;
- arbitrary six-degree-of-freedom grasping;
- deformable objects;
- a full rigid-body contact solver;
- a physical robot driver;
- a new Godel-Kolmogorov machine architecture;
- a new reward-shaping system visible to the agent.

A future MuJoCo or physical backend may implement the same world interface, but the v1 environment must remain fast, deterministic, inspectable, and macOS-friendly.

---

## 3. Why this must use an ARC-style contract first

Expose this standalone Python protocol:

```python
env.reset() -> np.ndarray
env.frame() -> np.ndarray
env.telemetry() -> dict[str, object]
env.step(action) -> np.ndarray
env.clone() -> env
env.levels_completed -> int
env.terminal() -> bool
env.actions -> tuple[int, ...]
```

Use this contract so the new domain exercises the same observation, experimentation, program growth, and replay machinery as the current ARC-AGI-3 work.

But distinguish **interface opacity** from **world uncertainty**:

- Raw integer action IDs are an API encoding, not a scientific mystery.
- The proposer prompt/runtime documentation must state their exact meanings.
- The separate telemetry packet must make the active axis, command, measured
  pose, gripper state, load, and rejection status explicit.
- The hidden content is what happens to objects and the scene after the known command.

Do not bypass the frame/action/reward contract with direct object-state
dictionaries. Conversely, do not waste levels and model budget forcing the
agent to infer arbitrary button mappings that would be calibrated or documented
on a real robot.

A normal standalone call should work:

```python
from roboarm_game import make_env

env = make_env("rb01-v1", seed=0)
frame = env.reset()
```

The custom environment therefore has ARC-style operational semantics without
being an ARC API game. It scientifically represents a known robot tool acting on
an initially unknown world.

## 4. Architectural overview

Use five conceptual layers.

```text
┌─────────────────────────────────────────────────────────┐
│ Godel-Kolmogorov machine proposer                       │
│ sees RGB camera + telemetry + actions + sparse reward   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│ Standalone RoboArmEnv facade                            │
│ validates integer actions and owns reset/step/clone     │
│ returns synchronized camera/telemetry and advances      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│ RoboArmWorld                                             │
│ state, kinematics, collision, grasp, release, goals      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│ Standalone deterministic RGB sensor                      │
│ perspective camera + occlusion -> 128×72×3 RGB8          │
└─────────────────────────────────────────────────────────┘
                           │ same world snapshot
┌──────────────────────────▼──────────────────────────────┐
│ Browser attempt replay viewer                            │
│ exact RGB/telemetry + larger 3D state reconstruction    │
└─────────────────────────────────────────────────────────┘
```

### Recommended source layout

```text
roboarm/
├── pyproject.toml
├── src/
│   └── roboarm_game/
│       ├── __init__.py
│       ├── config.py
│       ├── state.py
│       ├── kinematics.py
│       ├── geometry.py
│       ├── dynamics.py
│       ├── interface.py          # public documented control contract
│       ├── environment.py        # standalone public environment facade
│       ├── protocol.py           # structural typing contract
│       ├── observation.py        # RGB/telemetry schema and validation
│       ├── render_world.py       # deterministic perspective camera
│       ├── render.py             # unscored Phase-0 calibration only
│       ├── levels.py
│       ├── gkm/
│       │   ├── scenario.py       # closed untrusted proposal schema
│       │   ├── arena.py          # host-only preflight/commit connector
│       │   ├── safety_fsa.py     # deterministic admission state machine
│       │   ├── workspace.py      # proposal-only Codex workspace
│       │   ├── runner.py         # propose/FSA/feedback/replay/promote
│       │   ├── replay.py         # host-owned fresh replay
│       │   ├── accounting.py
│       │   └── taint.py
│       ├── oracle.py
│       ├── manual_play.py
│       └── README.md
├── tests/
│   ├── test_kinematics.py
│   ├── test_dynamics.py
│   ├── test_render.py
│   ├── test_game.py
│   └── test_gkm_integration.py
├── web/
│   ├── .openai/hosting.json
│   ├── app/
│   ├── components/
│   ├── lib/                    # replay-artifact/view-state bridge
│   ├── public/
│   └── tests/
├── references/
└── artifacts/
```

This layout is mandatory. In particular, do not create an ARC wrapper or local
ARC environment tree. The complete environment belongs in the importable,
testable `roboarm_game` package.

`interface.py` and the relevant README section are public apparatus
documentation: action meanings, coordinate conventions, step sizes, camera
calibration, telemetry schema, and rejection semantics. `dynamics.py`,
collision/grasp internals, level generation, and hidden thresholds are private
benchmark implementation.

Block the simulator dynamics and level-generation implementation from proposer access, but explicitly provide the actuator interface contract in the prompt/runtime documentation. The solver must discover object and scene mechanics by interaction; it is not required to reverse-engineer the robot controller.

---

## 5. Standalone environment facade

### 5.1 Full-frame output

The simulator renders and returns a complete `128×72×3` RGB8 frame itself. There is no
sprite layer, game engine, or external rendering lifecycle.

Each public step is:

1. Receive and validate one integer action.
2. Map it to a documented simulator command.
3. Advance `RoboArmWorld` by one quasi-static turn.
4. Render `RoboArmWorld` to a `128×72×3` RGB8 array.
5. Project the same action boundary into a separate public controller packet.
6. Check the level predicate and advance or terminate if appropriate.
7. Return owned copies of the frame and, through `telemetry()`, the packet.

### 5.2 Environment class

The facade should look conceptually like:

```python
class RoboArmEnv:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._world = RoboArmWorld(...)
        self._frame = np.zeros((72, 128, 3), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        self._world.reset(level_index=0, seed=self._seed)
        return self._render()

    def step(self, action: int) -> np.ndarray:
        sim_action = ACTION_MAP[action]
        self._world.step(sim_action)
        if self._world.level_won:
            self._advance_level()
        return self._render()

    def telemetry(self) -> dict[str, object]:
        return copy.deepcopy(self._public_controller_packet())

    def clone(self) -> "RoboArmEnv":
        return copy.deepcopy(self)
```

`frame()` returns a defensive copy. `clone()` returns an independent exact copy.
The public facade owns `levels_completed`, `terminal()`, and the immutable
`actions` tuple. No package outside NumPy is needed for this protocol.

### 5.3 Identity and baselines

Expose `game_id == "rb01-v1"` directly from the environment. Store versioned
configuration and eventual oracle baseline counts in ordinary package data, not
in ARC metadata.

---

## 6. External action protocol

### 6.1 Default: documented end-effector controller

The default benchmark must not use concealed joint semantics. Supply the
following action contract verbatim to the Godel-Kolmogorov machine proposer and
document it in the game README.

Use six coordinate-free integer actions:

| Action ID | Documented meaning |
|---|---|
| `ACTION1` | Decrease the currently selected command coordinate by one step |
| `ACTION2` | Increase the currently selected command coordinate by one step |
| `ACTION3` | Select the previous command coordinate |
| `ACTION4` | Select the next command coordinate |
| `ACTION5` | Open the gripper |
| `ACTION6` | Close the gripper |

The selected command coordinate cycles through:

```text
AZIMUTH → REACH → HEIGHT → AZIMUTH
```

Interpretation:

- `AZIMUTH`: base-centered horizontal angle of the gripper target;
- `REACH`: radial distance of the gripper target from the base axis;
- `HEIGHT`: gripper target height above the table.

The agent is told this exact convention. It does not need to infer it.

Canonical disclosed step sizes, expressed in physical units:

```python
AZIMUTH_STEP_DEG = 5.0
REACH_STEP_M = 0.020
HEIGHT_STEP_M = 0.015
CONTROL_TURN_S = 0.25
```

These increments are intentionally larger than the arm's listed repeatability and small relative to its roughly half-metre radial reach. They permit useful contact localization without turning the game into hundreds of microscopic moves. They live in configuration and in the interface contract. Changing them between evaluation seeds is prohibited.

### 6.2 Deterministic articulated controller

Each coordinate command updates a commanded end-effector target. A deterministic inverse-kinematics controller maps that target to the RoArm-M2-S base/shoulder/elbow chain using one fixed elbow branch and one documented tie-breaking rule.

The simulator still performs genuine articulated checks:

1. update the commanded target;
2. solve the configured IK branch;
3. generate the candidate articulated pose;
4. test joint limits, table collision, self-collision, obstacle collision, and attached-object collision;
5. accept the complete turn or reject it.

Thus the arm remains physically constraining, but the controller is not the puzzle.

### 6.3 Rejected commands

If the commanded target is unreachable or the candidate articulated motion is unsafe:

- do not apply the pose change;
- consume the action;
- keep the commanded coordinate at its previous accepted value;
- report rejection/load in the synchronized controller packet;
- leave the object scene unchanged unless contact occurred along an explicitly simulated swept path.

The interface documentation must state that commands can be rejected for reachability or collision. Exact collision geometry remains hidden.

### 6.4 Separate controller telemetry

The frame is camera-only. A synchronized structured packet must expose:

- selected command coordinate and last action;
- current commanded azimuth, reach, and height;
- measured joint positions and TCP position;
- gripper open/closed state and aperture;
- normalized contact load;
- motion rejection and controller interlock reason;
- turn and simulation timestamp.

These fields are known apparatus data, not clues to be decoded through trial
and error. The packet must not expose object coordinates, attachment flags,
support identity, target predicates, private mechanics events, or a success
recipe.

### 6.5 Optional low-level embodiment ablation

A separate non-default mode may expose direct joint increments and conceal or disclose them for an embodiment-learning ablation. It must not be mixed into the principal result and must not determine the canonical curriculum.

## 7. Hidden world state

Use plain dataclasses and NumPy arrays. Everything must be `copy.deepcopy` safe.

A minimum state:

```python
@dataclass
class RobotState:
    base_yaw: float
    shoulder_pitch: float
    elbow_pitch: float
    selected_axis: int
    command_azimuth: float
    command_reach: float
    command_height: float
    gripper_open: bool
    gripper_aperture: float
    contact_load: float
    last_motion_rejected: bool

@dataclass
class ObjectState:
    object_id: str
    shape: str
    position: np.ndarray       # x, y, z
    yaw: float
    size: np.ndarray           # width, depth, height
    mass: float
    friction: float
    graspable: bool
    fragile: bool
    broken: bool
    color_role: int

@dataclass
class WorldState:
    robot: RobotState
    objects: list[ObjectState]
    obstacles: list[Obstacle]
    supports: list[SupportSurface]
    attached_object_id: str | None
    attached_relative_pose: ...
    action_count: int
    level_won: bool
    level_failed: bool
    rng_state: ...
```

Do not store rendering objects, open handles, UI resources, threads, callbacks, or non-copyable physics-engine state in `WorldState`.

---

## 8. RoArm-M2-S kinematic and actuation model

Do not implement an arbitrary planar arm. Implement a lightweight model calibrated to the official Waveshare RoArm-M2 Xacro.

### 8.1 Joint structure

The canonical robot state has:

```text
q0  base yaw
q1  shoulder pitch
q2  elbow pitch
qg  gripper opening command
```

The physical product uses five servos because the shoulder is dual-driven, but the paired shoulder servos represent one commanded joint. Do not create a fifth independent action dimension.

Use the official software limits:

```python
Q0_LIMITS = (-3.1416, 3.1416)
Q1_LIMITS = (-1.5708, 1.5708)
Q2_LIMITS = (-1.0, 2.95)
QG_LIMITS = (0.0, 1.5)
```

The gripper command is continuous internally even though the canonical public actions request open and close. A valid grasp may stop closure early when both jaws contact an object.

### 8.2 Exact transform chain

Implement one direct reference function using 4x4 homogeneous transforms matching `roarm_m2.xacro`. At minimum include:

```text
world -> base_link                    fixed
base_link -> link1                    yaw q0, origin (0.0100000009, 0, 0.1230592705)
link1 -> link2                        pitch q1 with Xacro fixed rotation (-pi/2, -pi/2, 0)
link2 -> link3                        pitch q2, origin (0.2368151329, 0.0300023995, 0), fixed yaw +pi/2
link3 -> gripper_link                 gripper qg, origin (0.002906, -0.21599, -0.00066683), fixed rotation (-pi/2, 0, -pi/2)
link3 -> hand_tcp                     fixed, origin (0.002, -0.2802, 0), fixed rotation (+pi/2, 0, -pi/2)
```

The transform implementation is the ground truth for:

- joint and TCP positions;
- gripper orientation;
- link segment endpoints;
- camera rendering anchors;
- collision primitive placement;
- numerical tests.

### 8.3 Reduced fast model

The runtime may use an analytic cylindrical reduction for inverse kinematics and planning. Derive its effective dimensions from the Xacro rather than inventing them:

```python
SHOULDER_HEIGHT_M = 0.123059270461044
BASE_AXIS_X_OFFSET_M = 0.0100000008759151
UPPER_ARM_EFFECTIVE_M = hypot(0.236815132922094, 0.0300023995170449)  # about 0.2387 m
FOREARM_TO_TCP_EFFECTIVE_M = 0.2802
```

The reduced model must agree with the direct transform chain to a configured tolerance over the canonical command lattice. Where the reduction and exact chain differ near boundaries, exact-chain legality wins.

### 8.4 Known command-space controller

The public controller operates in cylindrical TCP target coordinates:

```text
(azimuth, reach, height)
```

The supplied interface defines these relative to the base yaw axis and tabletop. For each candidate target:

1. solve the documented fixed IK branch in the reduced model;
2. map the solution into the exact Xacro joint conventions;
3. evaluate the exact transform chain;
4. refine deterministically if TCP error exceeds tolerance;
5. interpolate the accepted joint path using bounded joint velocity;
6. test swept robot, object, table, and obstacle collision;
7. accept or visibly reject the complete command.

The controller convention, axis definitions, increments, and branch policy are supplied to the agent. There is no scientific credit for rediscovering them.

### 8.5 Workspace validation

Generate a sampled reachable-workspace test from the legal joint ranges. It must satisfy all of the following:

- approximately rotationally symmetric about the base axis when obstacles are absent;
- maximum horizontal diameter consistent with the published `1.090 m` envelope within a documented modeling tolerance;
- maximum vertical extent consistent with the published `0.798 m` envelope within a documented modeling tolerance;
- no generated canonical level requires a TCP target outside the validated workspace;
- table placement and base height are explicit, not silently tuned per level.

The exact extrema need not equal marketing values to the millimetre because the product envelope may include tooling and conventions not identical to `hand_tcp`. Any discrepancy greater than 5% must be documented and investigated rather than hidden by arbitrary scale factors.

### 8.6 Motion timing and interpolation

Use the listed no-load servo speed of `40 rpm` as an upper bound:

```python
SERVO_NO_LOAD_MAX_RAD_S = 40.0 * 2.0 * pi / 60.0  # about 4.19 rad/s
CONTROL_TURN_S = 0.25
```

A turn may require multiple deterministic substeps. Use a load-dependent speed multiplier in realism modes, but keep canonical clean mode deterministic. Never teleport the robot from one endpoint pose to another when that would skip contact or collision.

### 8.7 Payload and load envelope

Represent the advertised payload as a conservative quasi-static moment envelope:

```python
RATED_PAYLOAD_KG = 0.5
RATED_PAYLOAD_REACH_M = 0.5
RATED_PAYLOAD_MOMENT_NM = RATED_PAYLOAD_KG * 9.80665 * RATED_PAYLOAD_REACH_M
```

For held objects, compute a deterministic normalized load estimate from object mass, horizontal lever arm, robot posture, and optional acceleration. Use it for:

- visible load telemetry;
- slower motion in realism mode;
- grasp slip probability or deterministic slip threshold in advanced levels;
- rejection of clearly impossible lifts;
- parity with later physical current/load feedback.

Do not use this as a high-fidelity servo-torque model. The shoulder is dual-driven and the listed servo torque does not directly translate into a simple whole-arm limit. The advertised `0.5 kg at 0.5 m` envelope is the safer benchmark-level reference.

Canonical curriculum objects should usually be `0.02-0.15 kg`, leaving a large safety margin. Heavy-object transfer levels may approach the envelope deliberately.

### 8.8 Repeatability and encoder realism

Canonical clean mode remains exact and deterministic. Add optional seeded realism parameters based on the product specifications:

```python
ENCODER_COUNTS_PER_REV = 4096
REPEATABILITY_SIGMA_M = 0.0            # clean mode
REPEATABILITY_MAX_M = 0.004            # realism envelope
JOINT_QUANTIZATION_ENABLED = False      # optional ablation
```

When enabled, endpoint error must be seeded and replayable. Do not inject unseeded noise. Quantization and endpoint error should affect the robot pose and rendered scene, not merely decorate telemetry.

### 8.9 Hardware fidelity policy

The simulator is not a digital twin. The required fidelity is:

- exact published joint topology, origins, frame rotations, TCP, and software limits;
- realistic workspace scale in metres;
- fitted link/jaw collision geometry;
- bounded speed and swept motion;
- plausible payload/load behavior;
- optional repeatability and encoder effects;
- deterministic replay and macOS-friendly execution.

Do not add motor temperature, voltage sag, Wi-Fi latency, or flexible-link dynamics until object-contact learning works and a specific experiment requires them.

### 8.10 Controller guarantees

Required behavior:

- joint state remains within official configured limits;
- direct and reduced kinematics are numerically stable at boundaries;
- accepted command changes produce deterministic TCP displacement;
- rejected commands are visibly indicated and leave accepted command state unchanged;
- the same command from the same state produces the same articulated trajectory;
- level generation never requires unreachable poses;
- every accepted path is checked at deterministic swept substeps;
- direct-transform and reduced-model parity tests pass at sampled legal states.

## 9. Geometry and collision

### 9.1 Collision primitives

Use simple deterministic primitives fitted to the official RoArm-M2 mesh extents and transform chain:

- base: cylinder or low convex prism;
- shoulder housing: capsule/box compound;
- upper arm and forearm: capsules aligned to Xacro link frames;
- gripper palm: oriented box;
- gripper jaws: two oriented boxes driven by the gripper command;
- object: oriented box or vertical cylinder;
- obstacle: axis-aligned/yaw-oriented box or cylinder, including every visible
  cap, rim, or overhang;
- target receptacle: support floor plus each visible wall/rim as separate
  collision solids;
- table/support: horizontal rectangle at a fixed height.
- fixed workcell: base pedestal, safety posts, and rear/side wall solids where
  they are visible in the camera or browser reconstruction.

The first implementation may approximate oriented-box collision conservatively using projected intervals or sampled surface points. It must not permit obvious link-through-table or jaw-through-object artifacts.

The authoritative mechanics, CPU camera, and browser reconstruction must derive
these bodies from the same dimensions and transforms. A decorative visible
solid may not be absent from collision, and an invisible conservative collider
may not make an apparently clear replay fail without an explanatory display.

### 9.2 Motion validation

For every candidate joint motion:

1. compute the candidate robot pose;
2. compute link and gripper geometry;
3. test robot self, base, table, obstacle body/cap, target floor/walls, and
   fixed-workcell collision;
4. test attached-object collision;
5. reject or accept the entire turn.

Do not resolve arm collisions by teleporting penetrated objects away.
Sweep gripper aperture changes as well as arm-joint changes. A rejected command
must atomically preserve the last legal full configuration, so neither the
authoritative camera nor the browser may display the proposed penetrating pose.

### 9.3 Object pushing

Implement limited planar pushing, but keep it subordinate to grasping.

When an unattached object on a support is contacted laterally by a gripper jaw or wrist:

- estimate the end-effector displacement over the turn;
- move the object by a friction-scaled fraction of that displacement;
- reject or truncate the push if the object would enter an obstacle or leave the support;
- never lift an object through pushing.

This creates realistic accidental interaction and prevents the game from being a pure “teleport to grasp point” puzzle.

A level that requires crossing a low barrier must be included so that pushing alone cannot solve the full curriculum.

---

## 10. Grasp mechanics

Do not use a single distance threshold that attaches the nearest object.

A valid grasp requires a bilateral enclosure condition.

### 10.1 Candidate grasp volume

Define a grasp region between the two jaws, with:

- a center at the gripper;
- a closing axis;
- a jaw depth;
- a vertical tolerance;
- a maximum object width.

### 10.2 Closing sequence

When `ACTION6` closes the gripper:

1. reduce aperture until either:
   - fully closed, or
   - both jaws contact a graspable object;
2. identify objects intersecting the jaw sweep;
3. require opposing contact on both sides of the same object;
4. require sufficient overlap along jaw depth and height;
5. reject multi-object ambiguous grasps unless explicitly supported;
6. compute contact load;
7. attach only when enclosure and force constraints pass.

### 10.3 Attachment

After a valid grasp:

- store the object-to-gripper relative transform;
- move the object rigidly with the gripper on accepted arm motions;
- include the object in collision checks;
- expose attachment only through rendered co-motion and telemetry.

### 10.4 Slip

Add a deterministic slip model behind a feature flag.

A grasp stability score may depend on:

- object mass;
- enclosure depth;
- aperture margin;
- object width;
- motion increment;
- collision with an obstacle.

For v1 curriculum levels 1–8, configure objects so a properly centered grasp is stable. Use slip only in realism/transfer levels after the core game works.

### 10.5 Opening and release

When `ACTION5` opens the gripper:

- detach any held object;
- apply gravity;
- settle the object on the highest valid support below it;
- preserve horizontal pose unless a simple collision correction is necessary;
- mark the object lost if it falls outside all support bounds;
- detect placement in a target bin only after support settling.

---

## 11. Gravity and support

Use event-based gravity, not a continuous physics loop.

Gravity is evaluated:

- after release;
- after an accepted arm motion carrying an object if the grasp slips;
- after a pushed object loses support.

A released object falls vertically to:

1. the highest obstacle/platform top under its horizontal footprint;
2. a target tray/bin floor;
3. the table;
4. otherwise out of the workspace.

This is enough to support lift, transport, obstacle crossing, platform placement, and dropping without a heavyweight physics engine.

---

## 12. Camera, telemetry, and replay rendering

### 12.1 Authoritative camera contract

Every operational frame must be:

```python
np.ndarray(shape=(72, 128, 3), dtype=np.uint8)
```

Bytes are row-major RGB8. There is no palette-index mode, painted HUD, text,
alpha channel, semantic mask, object ID, success banner, or hidden RGB side
channel in the scored round.

The camera is part of the public apparatus and has a versioned calibration:

```text
projection          pinhole
vertical field      40 degrees
position metres     (0.72, -0.10, 0.50)
target metres       (0.14, 0.07, 0.13)
world up            (0, 0, 1)
near/far metres     0.01 / 5.0
distortion          none
clean noise mode    deterministic
```

Any later camera/noise mode must be explicitly versioned and seeded. It may
not silently change the canonical clean observation.

### 12.2 Operationally faithful visual content

The exact camera must derive from the same authoritative state used for
mechanics and show:

- perspective scale and occlusion;
- the table/workspace and fixed room context;
- articulated base, links, joints, palm, and driven opposing jaws;
- movable objects, barrier/support geometry, and the target bin;
- object height and carried-object clearance;
- plausible material colors, surface normals, directional lighting, contact
  shadows, and fixed optical vignetting.

Rendering may use a deterministic CPU ray/z-buffer model and fitted primitives.
At 128×72 the goal is a plausible 16:9 C920s-derived observation with correct
geometry and occlusion, not a photorealistic or sim-to-real claim. The image must change
only because camera-visible authoritative state changes. A rejected command
may therefore leave camera pixels unchanged while telemetry records the
rejection.

### 12.3 Separate reference-device I/O packet

`env.telemetry()` returns a structured packet pairing independently timestamped
RoArm and C920s products at the host. It contains:

- sensor-contract version, host sequence, request/response/capture timestamps,
  and arm/camera skew;
- host-selected coordinate, last action, T=104 command, and generic interlock;
- stock-style T=1051 encoder angles, firmware-derived XYZ in millimetres,
  signed servo loads, torque-switch flags, and supply voltage;
- C920s 1080p/30 UVC source metadata and 128×72 RGB8 processed-frame metadata.

This is device and controller instrumentation, not private puzzle state. It
must not invent metric jaw aperture, TCP force, a collision-reason sensor, or
hardware synchronization. It must not contain object coordinates,
object/target identities, attachment flags, support identity, target
predicates, private mechanics events, FSA verdicts, or a recommended action.

### 12.4 Evidence binding

The connector records and hashes camera bytes and telemetry separately at
reset and after every action. An admitted preflight, one-use commit permit,
stepwise commit interlock, fresh-source verification, exact replay, browser
export, and public evidence projection must all bind both hashes. A legacy
 4096-byte indexed frame, old square RGB frame, or trace without telemetry is an obsolete schema and
must fail admission rather than load as a fallback.

### 12.5 Browser attempt replay viewer — normative

Provide a browser experience that makes actual Godel-Kolmogorov machine
attempts directly inspectable. Its scientific input is a saved attempt artifact
emitted by the trusted Python campaign host. The artifact binds the sensor
contract, scenario version, seed, proposer generation, candidate-source digest,
trace role, integer actions, RGB and telemetry hashes, sparse rewards,
disposition, and—when applicable—fresh replay and promotion receipt.

The viewer must offer at least one genuine failed attempt and one independently
replay-promoted success. It displays:

- a nearest-neighbor inset of the exact stored 128×72×3 RGB bytes supplied to
  the machine;
- the exact stored companion controller packet;
- a larger, explicitly labeled state-synchronized 3D human replay view;
- physically coherent articulated links and driven gripper jaws;
- table, bin, object, barrier, shadows, occlusion, and depth cues;
- smooth display interpolation between recorded turn boundaries;
- action/reward timeline, failure/rejection/success disposition, trace role,
  proposer generation, and promotion metadata.

Interactive orbit controls must remain on the open inspection side of the
workcell. Bound azimuth and camera distance and disable or constrain panning so
a rear/side safety wall cannot move in front of and obscure the robot. These
human-view restrictions do not alter the recorded machine camera.

The larger 3D view is explanatory output. It is not called the machine's camera
input and is never fed back to the proposer. Host-only visual snapshots used
to reconstruct it are removed from proposer-visible evidence.

Every displayed event must arise from an authoritative simulator transition:
the object moves only through contact or valid attachment; attachment requires
bilateral enclosure; collisions reject motion; release invokes
gravity/support; and success follows the same sparse level predicate.

The canonical “pick, clear an obstacle, carry, release, and settle” route may
remain under a clearly separated developer mechanics-test mode. It must never
be labeled as discovery, an LLM attempt, a machine solve, or campaign evidence.

Optional photographic textures, generated sprites, ambient imagery, or an
opt-in local webcam background may improve presentation. They must never
substitute for articulated geometry, alter evidence bytes, or encode a canned
outcome.

The browser may not send actions into the campaign, run a solver, repair
candidate code, choose a policy, author a verdict, or substitute a second
scientific mechanics model or hand-authored outcome animation.

## 13. Level curriculum

Implement a coherent **object-response curriculum** rather than a curriculum for decoding the robot.

The arm-only calibration scene is required for testing and manual familiarization but is not counted as a scored scientific level.

### Calibration scene — Known actuator check, unscored

**Scene:** empty table with several reachable pose beacons.
**Goal:** automated tests and a human operator verify that documented azimuth/reach/height commands move the gripper as specified.
**Purpose:** validate the interface, IK branch, collision rejection, renderer, and physical-adapter parity.

The oracle may use this scene, but Godel-Kolmogorov machine performance on it is not part of the headline acquisition result.

### Level 1 — Push response

**Scene:** one free block and a broad target zone on the same support.
**Goal:** move the block into the target without grasping.
**Purpose:** discover how approach direction, contact, swept motion, and friction transform the object scene.

The gripper starts open. The level should admit more than one valid contact trajectory.

### Level 2 — Push around obstruction

**Scene:** one block, a wall or fixed obstacle, and a target that cannot be reached by pushing from the initial side.
**Goal:** reposition the gripper and push from a useful side.
**Purpose:** learn that object response depends on contact geometry and environmental constraints, not merely command direction.

### Level 3 — Bilateral enclosure

**Scene:** one large graspable cube.
**Goal:** close the gripper around the cube and achieve a valid bilateral grasp.
**Purpose:** infer enclosure, opposing contact, aperture, and grasp preconditions.

Mere proximity or one-jaw contact must not complete the level.

### Level 4 — Attachment and lift

**Scene:** one graspable cube and a visible clearance marker.
**Goal:** grasp and raise the cube above the marker for at least one completed turn.
**Purpose:** learn attachment, object–gripper co-motion, gravity suppression while held, and lift clearance.

### Level 5 — Release and settlement

**Scene:** a graspable cube and nearby tray.
**Goal:** lift, transport, open the gripper, and leave the cube settled inside the tray.
**Purpose:** compose approach, grasp, attachment, transport, release, fall, and support.

### Level 6 — Carry over a barrier

**Scene:** a low wall between object and target.
**Goal:** lift high enough, carry over the wall, and release in the target.
**Purpose:** distinguish carried-object collision and height-dependent clearance from free-space gripper motion.

### Level 7 — Shape-conditioned grasp

**Scene:** an object whose width, depth, or yaw makes some closures fail.
**Goal:** achieve a stable grasp and place it.
**Purpose:** learn that grasp success depends on scene geometry and approach, not a fixed coordinate macro.

### Level 8 — Clutter and occlusion

**Scene:** one target object, one distractor or occluder, and a constrained approach corridor.
**Goal:** manipulate the target without losing it or displacing the distractor beyond tolerance.
**Purpose:** test persistence, partial observability, collision-aware approach, and verification after occlusion.

### Level 9 — Select the correct object

**Scene:** two manipulable objects and a target whose visual relation indicates which object belongs there.
**Goal:** place only the matching object.
**Purpose:** combine scene interpretation with learned manipulation mechanics.

Use a nonlinguistic relation such as matching color, shape, or texture role.

### Level 10 — Dynamics and geometry transfer

**Scene:** held-out object dimensions, orientation, initial position, friction regime, and mild clutter.
**Goal:** complete pick-and-place without modifying retained solver code.
**Purpose:** test whether the Godel-Kolmogorov machine retained reusable object-response transducers rather than a literal trajectory.

### Optional Level 11 — Marginal grasp stability

**Scene:** an object that can slip under shallow enclosure or aggressive motion.
**Goal:** establish a deeper grasp and move within a stable envelope.
**Purpose:** make grasp quality and motion-conditioned slip causally relevant.

Do not add this until deterministic grasping and replay are stable.

### Optional Level 12 — Ordered two-object manipulation

**Scene:** two objects and two target slots, with one object initially blocking access to the other.
**Goal:** manipulate both in the feasible order.
**Purpose:** test repeated macro reuse and longer-horizon scene transformation.

Across all scored levels, the action mapping, controller, camera calibration,
visual grammar, and telemetry schema remain fixed and disclosed. New difficulty
must arise from the scene, not from remapping controls.

## 14. Procedural variation and held-out evaluation

The published level path should be deterministic for `seed=0`, enabling exact replay.

Also support held-out seeds that vary:

- object position;
- target position;
- object width/height;
- obstacle placement within solvable bounds;
- camera perturbation by a few pixels/degrees;
- nonstructural material-color assignments;
- friction/load parameters;
- small actuation noise in realism mode.

Every generated instance must be checked for reachability by the oracle before use.

Separate two notions:

1. **Canonical replay:** exact deterministic replay on the canonical seed.
2. **Generalization:** success across held-out seeds using the same retained solver code.

Do not silently mix held-out seed results into the canonical replay claim.

---

## 15. Reward, termination, and action budgets

### 15.1 Reward

Expose only sparse level completion:

```text
levels_completed += 1
```

No dense reward, distance reward, grasp reward, or object-coordinate reward may be returned to the agent.

The generated solver is allowed to invent its own dense visual progress measures from frames, exactly as in the current raw Godel-Kolmogorov machine setup.

### 15.2 Action budgets

Use per-level action budgets large enough for exploration but small enough to punish uncontrolled wandering.

Suggested initial budgets:

```text
L1–L2: 100 actions
L3–L5: 140 actions
L6–L8: 220 actions
L9–L10: 280 actions
```

Tune only after manual/oracle baselines exist.

### 15.3 Failure

A level may fail or reset on:

- object irrecoverably leaving the table;
- fragile object breaking;
- action budget exhaustion;
- unrecoverable robot pose, if such a state exists.

Ordinary rejected motions and failed grasp attempts should not immediately terminate the level.

---

## 16. Determinism, cloning, and replay

The canonical simulator must be exactly replayable.

Requirements:

- same seed + same action sequence = same frames and terminal state;
- `copy.deepcopy(world)` yields an exact independent clone;
- cloning does not share RNG state mutably;
- replay validation uses a fresh environment;
- no wall-clock time, thread scheduling, global RNG, or unordered iteration may affect state.

The public environment clones the complete simulation state. Therefore the
entire `RoboArmEnv` object must be deepcopy-safe.

Add a property test:

```python
env_a = make_env("rb01-v1", seed=0)
env_b = env_a.clone()

for action in sequence:
    assert np.array_equal(env_a.step(action), env_b.step(action))
```

Also test divergence after taking different actions from cloned states.

---

## 17. Oracle controller

Implement an oracle controller for environment validation only.

The oracle may access hidden state and should:

- solve every canonical level;
- validate generated seeds;
- produce an approximate action baseline;
- diagnose unreachable level configurations;
- never be imported into solver workspaces;
- never be exposed through the public environment facade.

The oracle should plan through the same documented command lattice available to the agent. It may inspect hidden state for validation, but it must not bypass the controller by directly setting joint angles or teleporting objects.

Recommended structure:

1. discrete pose planner over `(azimuth, reach, height, gripper_state)`;
2. the same deterministic IK, swept collision, and rejection rules used by the game;
3. hidden-state goal predicates for contact/pregrasp/lift/target validation;
4. A* or BFS over public actions;
5. fixed high-level candidate phases for open/approach/close/lift/carry/release;
6. replan after contact, attachment, obstruction, or release.

This tests the actual public action lattice while keeping arm kinematics out of the learned scientific target.

Use the oracle to generate `baseline_actions` and to establish that each level is solvable with margin.

---

## 18. Manual player and visual debugger

Provide a macOS-compatible manual player as a mechanics debugger. Keep it
separate from campaign evidence.

Minimum controls:

```text
A / D      previous / next command axis
W / S      decrease / increase selected coordinate
O          open gripper
C          close gripper
R          reset level
N          next debug level
```

The manual player should display:

- enlarged nearest-neighbor rendering of the 128×72×3 RGB frame;
- current commanded pose, solved hidden joint values, and object poses in a separate debug pane or console;
- collision/rejection reason;
- grasp state;
- level predicate status.

Debug information must never be embedded in the agent-facing frame or public
telemetry. Debug panes remain host-only.

A frame-dump mode should save PNGs or NumPy arrays for regression tests.

The downstream browser replay viewer adds:

- orbit and calibrated-camera views;
- attempt selection plus play/pause/single-turn/restart controls;
- distinct failed, rejected, clone-probe, committed, verification, and promoted
  evidence labels;
- synchronized exact stored 128×72×3 RGB frame, telemetry, event log, and replay
  timeline;
- a clear distinction between commanded and accepted motion; and
- evidence metadata containing seed, scenario version, actions, terminal
  predicate, proposer generation, candidate-source digest, disposition,
  promotion receipt, and renderer/mechanics versions.

---

## 19. Godel-Kolmogorov machine integration

### 19.1 First integration target

Run a real headless-Codex proposer inside a Godel-Kolmogorov machine
program-growth loop. The
proposer writes and revises executable `legs.py`, thin per-level `players.py`,
`solve.py`, and bounded declarative scenarios. Those programs are
proposal-only: they transform host-sealed public evidence into scenario JSON
and have no live simulator or connector access.

The trusted campaign host:

- constructs the authoritative Python simulator;
- seals the initial RGB camera/telemetry pair and later public observations;
- validates the closed proposal schema;
- executes isolated digital-twin experiments;
- applies the deterministic safety FSA;
- mints and consumes any single-use commit permit;
- owns sparse reward and terminal observations;
- verifies candidate source in a fresh workspace;
- independently replays the exact admitted action boundary; and
- computes description-length/action cost.

The first smoke test must be launched from the `roboarm` package:

```bash
python -m roboarm_game.gkm_runner \
  --game=rb01-v1 \
  --rounds=...
```

The host-only connector accepts preflight sequences from the FSA and committed
sequences only with an authentic one-use in-memory permit. It advances exactly
one RoboArm turn per admitted action and records the resulting defensive 128×72
frame. It owns sparse reward, terminal state, reset, exact clone construction,
preflight accounting, committed-action accounting, and action evidence.
Proposer programs never receive a `RoboArmWorld`, connector instance, socket,
token, scene dictionary, mechanics event, object coordinate, attachment flag,
or other hidden state.

`gkm_runner` must depend only on the `roboarm_game.protocol` surface. It may
follow the existing Godel-Kolmogorov machine verifier-driven algorithm and
artifact conventions as
read-only research references, but it must not import the ARC harness, ARC
packages, or private environment state.

Do not create a general robotics agent framework. Build the smallest
protocol-native experiment driver needed for propose, validate, preflight,
observe, fail, revise, FSA-authorize, retain, and fresh-replay promote.

The first real round begins from a zero-seed solver workspace. A supplied
handwritten solver, the canonical 63-action mechanics fixture, a prerecorded
model response, a mocked proposer, a deliberately staged failure sequence, or
replay-only execution is test infrastructure—not a Godel-Kolmogorov machine
acquisition run.

### 19.2 Required known-interface appendix

The proposer must be told, verbatim and before its first experiment:

- `ACTION1/2` decrease/increase the selected command coordinate;
- `ACTION3/4` select previous/next coordinate;
- coordinate order is azimuth, reach, height;
- `ACTION5` opens and `ACTION6` closes the gripper;
- exact coordinate step sizes;
- the exact camera calibration and separate telemetry fields for selected axis,
  commanded/measured pose, gripper state, load, and rejection;
- commands are deterministic but may be rejected by reachability or collision;
- objects persist, collide, may be pushed, enclosed, grasped, carried, released, and supported;
- unsupported objects fall.

Do not reveal:

- object coordinates;
- friction values;
- collision margins;
- bilateral-contact thresholds;
- grasp stability formula;
- slip thresholds;
- target predicates beyond what is visually presented;
- level sequence or generated seeds.

This is not robotics-specific cheating. It models the calibration manual and command API supplied with an actual arm.

The disclosure is about the known apparatus, not the puzzle solution. It must
not contain a successful coordinate sequence, level-specific target geometry,
recommended grasp pose, obstacle-clearance height, or a recipe for the round.

### 19.3 Experimental conditions

Run the following conditions explicitly.

#### Condition A — known actuator, generic world priors

Use the documented interface appendix and otherwise retain the current generic
world prompt. This is the principal condition.

#### Condition B — known actuator, minimal manipulation priors

Add only generic physical priors about persistence, contact, enclosure, support, and falling. This measures the value of embodied preconceptions without hiding the controller.

#### Optional Condition C — concealed embodiment ablation

Remove some or all controller documentation and/or expose direct joint actions. Treat this only as an ablation. Do not let it define the main benchmark or consume the canonical curriculum.

### 19.4 Isolation-preserving integration

No file in the existing Godel-Kolmogorov machine implementation may be changed
or imported as a runtime dependency.

Implement all integration behavior inside `roboarm`, including:

- constructing `rb01-v1` through `roboarm_game.make_env`;
- accepting any environment satisfying `roboarm_game.protocol.Environment`;
- injecting the documented interface appendix;
- advertising only integer actions `(1, 2, 3, 4, 5, 6)`;
- materializing an offline scenario authoring/validation harness instead of a
  connector client;
- denying all proposer Unix sockets, general network access, connector tokens,
  and live action handles;
- blocking `roboarm_game` dynamics, geometry, levels, oracle, environment source,
  and hidden runtime state from proposer access;
- leaving only the public interface contract, camera calibration, and telemetry
  schema visible;
- redirecting scratch workspaces, generated programs, logs, caches, replay
  artifacts, and campaign reports into `roboarm/artifacts/` or an explicitly
  controlled temporary directory;
- adding all replay and integration tests under `roboarm/tests/`.

The runner may reproduce generic Godel-Kolmogorov machine accounting and
verification behavior, but
it must not import `arc/`, register the game in a parent loader, or write into
parent output locations.

### 19.5 Propose–probe–fail–retain–replay discipline

The campaign host, not prompt prose, structurally enforces the
Godel-Kolmogorov machine loop:

1. Materialize a fresh, receipt-bound workspace containing only the public
   apparatus contract, closed scenario contract, host-sealed public evidence,
   generic perception helpers, and the admitted solver lineage. There is no
   connector client, socket, token, or action method in the payload.
2. Ask headless Codex to transform the sealed evidence into compact
   falsifiable `experiment` scenarios and, only when supported, a `candidate`,
   while changing executable retained proposal source.
3. After the Codex process and every child exit, validate the scenario JSON
   against an exact field/action/size contract. The model cannot write observed
   or verified fields.
4. Run each valid scenario through the deterministic FSA. Experiments execute
   only in an isolated authoritative preflight. Candidates also preflight;
   unsafe, known-rejected, incomplete, or non-goal candidates receive no
   commit permit.
5. Project only public frame/action/reward/terminal facts and a host FSA
   disposition into the next generation. Keep private mechanics events and
   precise safety telemetry in the host/browser evidence ledger.
6. Preserve each clean unsuccessful generation, its source delta, transcript,
   proposed scenario, preflight action evidence, and failure disposition. Do
   not invent failures after the fact.
7. Reject the complete generation if it attempts private runtime/source
   inspection, constructs an actuation channel, modifies sealed evidence,
   escapes the workspace, loses transcript evidence, or uses another
   prohibited channel. Reject malformed proposals before any simulator action.
   Tainted bytes cannot seed later learning.
8. Try already promoted legs before paying for another proposer turn. Require
   `players.py` to compose named `legs.py` capabilities so reuse is observable
   and source growth is priced.
9. Authorize a commit only after an earlier generation has produced a genuine
   nonempty failed observation and a later source revision proposes a complete
   safe goal-reaching candidate. Bind the one-use permit to proposal,
   preflight, policy, and source receipts; guard every committed step with a
   just-in-time clone comparison.
10. Run the final proposal source in a fresh host-owned process from the exact
   admitted parent, reproduce the same candidate, and pass it through a fresh
   connector/FSA. Trim the resulting path to the first acquisition boundary.
11. Independently replay both candidate source and exact action path from zero.
   Only then emit a promotion and allow the new solver source into the next
   round.

Maintain separate immutable ledgers for proposed, admitted/rejected,
connector-observed, proposer-visible, deterministically verified, and promoted
facts. Promotion accounting includes marginal retained-source growth and an
explicit literal-action-container cost. A memorized coordinate list can be
retained as an honest hypothesis or regression artifact, but it is not evidence
of a reusable object-response transducer unless unchanged code transfers to
declared held-out scenes.

### 19.6 Later cone/transducer work

After the raw coding-agent loop clears multiple levels, inspect whether retained code factors into reusable legs corresponding to:

- detect free-space versus contact response;
- estimate object displacement from gripper sweep and contact side;
- identify obstruction or support constraints;
- center jaws around an object;
- close until opposing contact;
- verify attachment by object–gripper co-motion;
- lift to clearance;
- detect carried-object collision;
- release and verify fall/settlement;
- re-identify an object after occlusion.

Do not count actuator-axis selection or inverse kinematics as the target learned mechanism in the principal analysis.

## 20. Complexity accounting and measurements

Keep the existing Godel-Kolmogorov machine free-energy/replay accounting for
the first run.

Record at least:

- canonical levels completed;
- canonical replay action count;
- total committed object-interaction actions, excluding unscored calibration;
- total isolated preflight/clone actions;
- proposed scenarios, schema rejections, FSA rejections, deferred commits, and
  issued/consumed permits;
- retained solver source size;
- marginal retained code growth per promoted level;
- direct calls to unchanged helper legs;
- held-out seed success;
- failures by category;
- proposer compute/cost if available;
- exact source and replay artifact for each promoted level.

Do not conflate:

- replay path length;
- object-mechanics discovery interaction count;
- clone/lookahead count;
- model compute;
- retained code complexity.

For this game, preflight/clone usage is especially important because a physical
arm cannot provide exact state forks. Report it separately from committed
actions, and never describe a preflight success as a physical or promoted
success.

---

## 21. Realism modes

Implement realism as controlled switches, not as an inseparable soup.

```python
@dataclass(frozen=True)
class RealismConfig:
    action_noise: float = 0.0
    backlash: float = 0.0
    camera_jitter: float = 0.0
    endpoint_error_max_m: float = 0.0
    joint_quantization: bool = False
    load_speed_reduction: float = 0.0
    material_color_jitter: bool = False
    load_noise: float = 0.0
    slip_enabled: bool = False
    partial_occlusion: bool = True
    pushing_enabled: bool = True
```

Required presets:

### `clean`

- deterministic;
- no actuation noise;
- no slip;
- fixed camera;
- exact telemetry;
- canonical replay mode.

### `transfer`

- seeded small camera/geometry/material-color variation;
- no stochastic replay-breaking behavior;
- held-out evaluation.

### `noisy`

- seeded action/load noise;
- backlash;
- slip;
- stronger occlusion;
- used only after the clean game is solved.

Every noisy run must still be reproducible for a fixed seed.

---

## 22. Performance requirements

The simulator exists to support many clone experiments. Avoid heavyweight dependencies and per-step allocations where practical.

Targets on a normal laptop CPU:

- at least 500 complete `clone + step + render` operations per second in a simple level;
- substantially faster hidden-state stepping when rendering is disabled in tests;
- no memory growth over 100,000 random steps;
- frame generation bounded and deterministic;
- no recursion proportional to action history.

Browser targets on a current desktop browser:

- interactive rendering at 30 FPS or better while replaying a saved attempt;
- no camera/render frame may mutate or advance authoritative mechanics state;
- one fixed-step evidence trajectory independent of display refresh rate;
- deterministic action/event replay across page reloads for the pinned browser
  version; and
- a first meaningful frame without requiring an external ARC service.

These are engineering targets, not publication claims. Measure and report actual performance.

Use NumPy where it simplifies geometry/rasterization, but do not build an opaque vectorized system that is impossible to audit.

---

## 23. Tests

### 23.1 Kinematics

- known home pose;
- endpoint continuity;
- symmetry under base yaw;
- joint-limit behavior;
- numerical bounds;
- link geometry follows endpoint geometry.

### 23.2 Collision

- arm cannot pass through table;
- attached object cannot pass through barrier;
- rejected motion leaves pose unchanged;
- contact telemetry spikes on rejection;
- clone state remains independent.

### 23.3 Grasp

- proximity alone does not attach;
- one-jaw contact does not attach;
- bilateral centered closure attaches;
- oversized object cannot be grasped;
- opening detaches;
- held object follows gripper rigidly;
- release settles on the correct support.

### 23.4 Rendering

- frame shape/dtype/range;
- perspective camera calibration and deterministic pixel hashes;
- deterministic golden frames for every canonical level;
- height changes visible under camera projection;
- object occludes correctly in at least representative cases;
- selected-coordinate telemetry changes only on selector actions and matches
  the documented axis order;
- camera bytes and telemetry remain separate and synchronized;
- legacy indexed/HUD frames fail the operational sensor schema.

### 23.5 Game lifecycle

- `make_env("rb01-v1", seed=0)` constructs the standalone environment;
- `reset()` produces the expected first frame;
- only the six documented actions `(1, 2, 3, 4, 5, 6)` are advertised;
- each level advances exactly once on its predicate;
- action budget produces terminal state;
- fresh replay reproduces oracle completion;
- the environment deep-copies cleanly.

### 23.6 Robustness

- 10,000 seeded random-action episodes without exceptions;
- no NaNs or out-of-bounds objects;
- every generated held-out instance passes oracle solvability;
- random policy does not accidentally clear a large fraction of levels;
- no one-action exploit clears nontrivial levels.

### 23.7 Godel-Kolmogorov machine integration

- a host-only protocol-native connector can reset, step, and clone `rb01-v1`,
  but the proposer payload contains no connector, socket, token, or direct
  action capability;
- every frame/action/reward supplied to later proposer code is a host-sealed
  public projection of an authoritative connector preflight or commit;
- `RoboArmEnv.clone()` behavior is exact;
- the proposer receives the exact documented control contract;
- the closed scenario schema rejects extra model-authored `passed`,
  `observedStatus`, reward, terminal, authorization, safety, and verdict fields;
- invalid action IDs, oversized scenarios, duplicate IDs, non-finite JSON,
  unsafe trajectory/load/collision preflights, forged/reused permits, and
  commit/preflight divergence fail closed;
- experiment scenarios and clone-only candidate success cannot commit;
- a tiny handwritten solver may clear the unscored calibration scene only as a
  connector test and is never counted as acquisition;
- a live LLM proposer starts from a zero-seed solver workspace, performs
  observable bounded scenario experiments, leaves at least one genuine failed
  hypothesis, consumes its sealed result in a later generation, revises
  executable source, and produces a safe candidate;
- fresh host verification and independent source/path replay accept the exact
  first acquisition boundary before promotion;
- proposed, FSA-admitted/rejected, observed, public-projected, verified, and
  promoted facts are stored in separate receipt-bound ledgers;
- retained legs, per-level composition, marginal source growth, literal path
  cost, proposer attempts, failed generations, committed actions, preflight
  actions, replay actions, and compute are recorded separately;
- source-access guards block dynamics and level-generation internals while leaving the interface contract available.
- a path audit proves that the test and campaign created no file outside
  `roboarm/` or their explicitly controlled temporary directory.
- a dependency/import audit proves that runtime code does not depend on
  `arc_agi`, `arcengine`, or parent `arc/` modules.

### 23.8 Browser replay evidence

- a browser can select and replay at least one genuine unsuccessful
  Godel-Kolmogorov machine attempt
  and one independently replay-promoted success;
- attempt metadata makes `committed`, `clone_probe`, and `verification` traces,
  plus `failed`, `rejected`, and `promoted` dispositions, unambiguous;
- RGB camera, articulated joint transforms, TCP overlays, telemetry, event log,
  and 128×72 inset remain synchronized at every turn boundary;
- browser and Python golden transform fixtures agree within declared numerical
  tolerances;
- browser replay from seed plus integer actions reproduces the same accepted,
  rejected, contact, attach, release, settle, and success event sequence;
- the object cannot move through a decorative animation path without mechanics
  events;
- resizing, orbiting, pausing, or changing display FPS does not change the
  simulation result;
- the browser cannot plan, repair source, invoke a proposer, choose solver
  actions, or write a promotion;
- any canonical or manual mechanics mode is visibly segregated as test-only and
  cannot masquerade as campaign evidence;
- the production build has no ARC API dependency and no required external
  tracking or camera permission; and
- both a failed and successful replay are visually inspected in a real browser
  at desktop and narrow viewport sizes, with evidence captured under
  `roboarm/artifacts/`.

---

## 24. Acceptance criteria

The milestone is complete only when all of the following are true:

1. `rb01-v1` constructs through the standalone `roboarm_game.make_env` factory
   with no ARC package, loader, registration, or metadata dependency.
2. The proposer receives a 128×72×3 RGB8 camera frame, synchronized controller
   telemetry, action availability, level reward, and the documented actuator
   interface contract.
3. No scored level requires discovering arbitrary action meanings or deriving robot inverse kinematics.
4. The game implements the unscored calibration scene and at least scored Levels 1–10 above.
5. The arm internally uses genuine coupled forward/inverse kinematics and swept articulated geometry.
6. Known command increments produce deterministic gripper motion or a
   documented telemetry rejection.
7. Collision, pushing, bilateral grasp, attachment, release, gravity, and support are implemented.
8. The frame is a calibrated fixed perspective RGB camera; controller telemetry
   is a separate synchronized packet.
9. Canonical replay is exact.
10. The entire game is deepcopy-safe.
11. An oracle solves every canonical scored level.
12. Held-out scenes are generated and oracle-validated.
13. A manual player works on macOS.
14. Unit, regression, fuzz, and Godel-Kolmogorov machine integration tests pass.
15. The protocol-native machine runner operates only through the standalone
    environment contract and does not import an ARC harness or create a general
    robotics agent framework.
16. Dynamics and level-generation source are blocked from proposer inspection, while the public interface contract is supplied.
17. A concise `roboarm/src/roboarm_game/README.md` documents commands, the known-actuator boundary, and current limitations.
18. A path-scoped audit proves that implementation, tests, campaigns, and
    generated artifacts made no change outside `roboarm/`.
19. A dependency and import audit proves that `arc_agi`, `arcengine`,
    `ARCBaseGame`, and parent `arc/` modules are absent from runtime code.
20. A genuine live headless-Codex Godel-Kolmogorov machine round starts from an admissible zero-seed
    proposal-only workspace with no connector authority, emits bounded
    scenarios, preserves an actual failed authoritative preflight, consumes
    that sealed evidence in a later retained-source revision, obtains an
    safety-FSA-authorized stepwise-interlocked committed success, and earns at least
    one exact-boundary promotion through independent fresh source/path replay.
    A mocked model, supplied solver, canonical mechanics path, clone-only
    success, model-authored verdict, or replay-only run does not pass.
21. A deployed or explicitly local browser viewer replays at least one genuine
    failed attempt and one promoted success. Its exact 128×72×3 RGB inset,
    companion public telemetry, larger human replay view, arm/object state,
    event log, sparse success predicate, and attempt evidence are synchronized
    to the same authoritative connector transitions; a canned animation does
    not pass.

A reach-only curriculum, concealed-button puzzle, magic-distance pickup,
two-level scaffold, separate Gym demo, scripted “discovery” controller,
canonical-path playback presented as learning, or visually polished but
mechanically fake browser animation does not satisfy this specification.

## 25. Implementation sequence

### Phase 1 — Standalone shell and known actuator

Deliver:

- package layout;
- standalone `rb01-v1` factory and environment facade;
- full-frame renderer;
- documented RGB camera calibration and controller-telemetry schema;
- cylindrical command-space controller;
- deterministic IK branch;
- accepted/rejected command behavior;
- unscored calibration scene;
- manual player;
- browser rendering shell used only for transform/frame parity and
  mechanics-test replay at this phase.

Exit criterion: the documented commands produce the expected gripper motion
through the standalone public environment; no learning is required to validate
this, no ARC dependency is imported, and no file outside `roboarm/` is changed.

### Phase 2 — Contact and pushing

Deliver:

- object boxes/cylinders;
- support surfaces;
- swept contact;
- deterministic pushing;
- obstruction and friction;
- Levels 1–2.
- replay-view parity for recorded contact events.

Exit criterion: the same known gripper command produces different object outcomes depending on contact side, friction, and obstacles.

### Phase 3 — Enclosure and grasp

Deliver:

- jaw geometry;
- opposing-contact detection;
- aperture constraints;
- contact/load telemetry;
- attachment;
- Levels 3–4;
- replay-view parity for bilateral-contact, attachment, and load events.

Exit criterion: only genuine bilateral enclosure can grasp, and lifting visibly verifies attachment.

### Phase 4 — Carry, collision, release, and support

Deliver:

- attached-object collision;
- gravity;
- support settling;
- target trays/bins;
- Levels 5–6;
- canonical mechanics regression fixture for pick, barrier clearance, carry,
  release, and stable target settlement.

Exit criterion: oracle and manual player can pick, carry over a barrier, release, and verify settlement.
The same test fixture must replay visibly with synchronized mechanics events
and the exact 128×72×3 RGB inset, clearly labeled as developer-test evidence.

### Phase 5 — Scene generalization

Deliver:

- Levels 7–10;
- held-out seeded generation;
- geometry, friction, clutter, and material-color variation;
- oracle solvability checks;
- baseline action counts;
- a versioned campaign replay-artifact schema.

Exit criterion: one retained solver can be evaluated unchanged across canonical
and held-out scene dynamics without relying on browser behavior.

### Phase 6 — Godel-Kolmogorov machine campaign integration

Deliver:

- a `roboarm`-owned CLI and test entry point;
- a host-only protocol-native connector accepting the standalone environment;
- a closed declarative scenario schema and public-evidence projection;
- a deterministic safety FSA with isolated preflight, explicit transitions,
  trajectory/load/collision/budget checks, one-use commit permits, and
  just-in-time per-action clone comparison;
- a real live-LLM coding proposer with a clean zero-seed first round;
- a Codex payload and sandbox containing no connector client, socket, token,
  live environment, or action method;
- the exact known-interface appendix;
- Conditions A and B;
- optional concealed-embodiment ablation flag;
- structurally enforced retained `legs.py` and thin per-level players;
- bounded and separately metered preflight probes;
- clean failed-generation/WIP preservation and tainted-generation quarantine;
- fresh source/path verification, exact-boundary replay, promotion receipts,
  marginal source and literal-action accounting;
- separate proposed, admitted/rejected, observed, public, verified, and
  promoted evidence ledgers;
- failed and successful replay artifact collection;
- redirection of every cache, scratch workspace, generated solver, log, replay,
  and report into `roboarm/artifacts/`;
- initial campaign report.

Exit criterion: headless Codex genuinely proposes experiments without actuation
authority, the host records a real failed preflight, a later Codex generation
changes executable proposal source from the sealed public outcome, and a safe
candidate earns an FSA-authorized commit plus independent exact-boundary
promotion. Held-out results and all costs are reported honestly, and a path
audit confirms no write outside `roboarm/`.

### Phase 6B — Browser attempt replay viewer

Only after genuine Phase-6 artifacts exist:

- list failed, rejected, and promoted proposer generations;
- replay committed, clone-probe, and verification traces with their roles
  visually distinct;
- show synchronized RGB, exact 128×72 input, action/reward timeline, and
  promotion metadata;
- visually inspect one genuine failed replay and one promoted success at
  desktop and narrow viewports.

Exit criterion: the browser only illustrates authoritative campaign evidence
and contains no planning, solver, repair, hidden-state, or promotion path.

### Phase 7 — Controlled realism

Only after Phase 6:

- backlash behind the known controller;
- action noise;
- telemetry noise;
- slip;
- stronger occlusion;
- optional camera perturbation.

Exit criterion: clean performance remains unchanged and each realism feature has an isolated ablation.

## 26. Coding-agent operating instructions

1. Inspect the current repository read-only only where it helps preserve
   Godel-Kolmogorov machine experimental semantics; do not use an ARC
   implementation API.
2. Create or modify files only below `/Users/sasha/gkm/roboarm`; never add,
   modify, delete, install, cache, or generate a file under `arc/`, the root
   `environment_files/`, or another sibling directory.
3. Preserve the existing Godel-Kolmogorov machine experiments as read-only
   research references.
4. Implement the game in small, testable modules.
5. Do not introduce ROS or a heavyweight simulator dependency.
6. Supply the exact actuator/control contract, but do not expose hidden object, grasp, contact-identity, or physics state to the solver.
7. Do not use a distance-only magic pickup.
8. Do not hard-code an oracle path into level logic.
9. Keep all randomness local and seeded.
10. Keep every runtime object deepcopy-safe.
11. Add tests with each mechanics layer.
12. All tests must target the standalone `roboarm_game` protocol and redirect
    cache output into `roboarm/artifacts/` or a controlled temporary directory.
13. Provide actual commands and observed results in the final implementation report.
14. Be explicit about any standalone protocol limitation that forces a design
    change inside the isolation boundary.
15. Prefer a complete deterministic mechanics slice over many half-implemented features.
16. Do not turn arm-control discovery into the benchmark; finish the playable object-response curriculum.
17. Treat the browser only as a downstream viewer of saved failed/successful
    Godel-Kolmogorov machine attempt evidence, never as a controller, solver, campaign input, or
    second source of mechanics.
18. Do not fake arm motion, object motion, attachment, release, collision, or
    success with a canned clip, spline, or sprite sequence.
19. Cross-check the browser’s joint transforms, exact 128×72×3 RGB frames, and mechanics
    event trace against the authoritative Python connector artifacts before
    presenting a replay.
20. Visually inspect one real failed attempt and one promoted success at desktop
    and narrow viewports, and save their seed, actions, event trace,
    proposer-generation identity, disposition, and promotion evidence below
    `roboarm/artifacts/`.
21. Give the proposer the complete documented action/controller contract, but
    no connector authority. Feed every frame, action result, preflight,
    terminal state, and sparse reward from the host-owned RoboArm connector
    through a sealed public-evidence projection; never substitute a browser,
    model-authored observation, or scripted transition.
22. Treat every proposer scenario as untrusted. Only the deterministic safety
    FSA may authorize a commit, and only through a single-use permit bound to a
    safe authoritative preflight. Never encode the task solution in the FSA.

---

## 27. Implemented verification commands

Run these commands from `roboarm/`:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q

cd web
npm test
npm run build
```

The operational environment is available through ``roboarm_game.make_env``;
there is no separate `manual_play` or `oracle` module in this release. The
production campaign command and report/viewer regeneration commands are kept in
`README.md`, where their complete current arguments are tested against the
implemented CLI. The runner places every generated output below
`roboarm/artifacts/` unless an explicitly validated in-project destination is
provided.

---

## 28. Future physical adapter

Do not implement this in v1, but preserve the boundary.

A future physical backend should present the same **known command-space protocol**:

```text
ACTION1/ACTION2  decrease/increase selected command coordinate
ACTION3/ACTION4  select azimuth/reach/height
ACTION5          open gripper
ACTION6          close gripper
```

The adapter should translate the documented command target into the physical RoArm's coordinate-control interface through an external calibrated controller. The product supports USB/UART and HTTP communication using JSON commands; the first macOS adapter should prefer direct USB serial or local HTTP and must not require ROS2. Calibration, servo mapping, safety limits, and low-level trajectory execution belong below the learning system.

Observation construction:

1. capture the fixed calibrated camera frame;
2. rectify, crop, and resize to 128×72 RGB8;
3. collect controller measurements for command, joints/TCP, gripper, load, and
   rejection;
4. timestamp and seal the image and telemetry as one synchronized boundary;
5. never paint telemetry into the camera pixels.

The Godel-Kolmogorov machine should not need to rediscover which motor command
moves which link. Simulated and physical experiments should differ mainly in
the object/scene response, noise, occlusion, backlash, and contact uncertainty.

The major incompatibility is cloning. Preserve clone use as a measured resource in simulation. A later physical experiment must either:

- run without exact clone;
- use a learned internal transducer as the clone model;
- or explicitly classify simulator lookahead as pretraining.

Do not hide this distinction.

## 29. Final scientific standard

A successful result is not:

> “The agent decoded which button bends which robot joint.”

Nor is it merely:

> “The agent eventually moved a colored square into a box.”

The intended evidence is:

- a live LLM proposer received the documented action nature but no private
  mechanics or solution trace;
- every proposer observation, action outcome, clone, and sparse reward came
  from the standalone RoboArm connector;
- unsuccessful hypotheses and code generations were preserved rather than
  scripted or narrated after the fact;
- the robot is a documented and calibrated instrument;
- the environment contains real articulated reachability and collision constraints;
- object response depends on contact side, swept motion, friction, obstruction, enclosure, and support;
- grasp and attachment have nontrivial observable preconditions;
- the solution composes pushing or approach, enclosure, lifting, transport, release, and verification;
- the same retained code handles later levels and held-out object/scene configurations;
- every promoted capability has a replay certificate;
- marginal code growth decreases when genuine object-response helper routines are reused;
- object-interaction actions, clone calls, replay actions, and compute are reported separately;
- arm-only calibration is excluded from the headline learning count.
- the browser only illustrates genuine failed and successful replay evidence
  after the campaign and contributes no solving behavior.

The concise scientific framing is:

> **Known actuator, unknown world dynamics.** The Godel-Kolmogorov machine is
> asked to learn the action–response structure of objects and scenes under
> manipulation, not to reverse-engineer the robot’s controller.

That makes `rb01` a meaningful embodied extension of the current ARC-AGI-3
Godel-Kolmogorov machine program rather than an arbitrary hidden-controls game
or a decorative robotics demo.

## 30. Hardware reference sources

The coding agent should record the exact revision or commit inspected when implementation begins.

1. Reichelt product page, article `WS-25974`, Waveshare RoArm-M2-S: published workspace, payload, accuracy, speed, ranges, feedback, communications, mass, and package contents.
   `https://www.reichelt.com/ch/de/shop/produkt/roboterarm_esp32_4dof_360_0_5_kg-392078`

2. Waveshare official RoArm-M2 source repository: controller and product documentation.
   `https://github.com/waveshareteam/roarm_m2`

3. Waveshare official ROS2/MoveIt workspace, branch `ros2-humble`: URDF/Xacro, meshes, joint limits, TCP, and command examples.
   `https://github.com/waveshareteam/roarm_ws`

4. Canonical Xacro path:
   `src/roarm_main/roarm_description/urdf/roarm_m2/roarm_m2.xacro`

Vendor specifications and the Xacro are reference inputs, not runtime dependencies. Copy the small set of required constants into a versioned local hardware profile with source comments and tests.
