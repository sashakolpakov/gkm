# rb01-v1 public apparatus contract

This document, `protocol.py`, and `interface.py` are the complete RoboArm
apparatus disclosure intended for a proposer. Campaign tooling may additionally
provide a closed scenario-authoring contract, host-sealed public evidence, and
generic perception helpers. It provides no connector client, socket, token,
environment handle, clone handle, action method, private simulator state,
fixtures, successful action trace, or safety verdict.

`rb01-v1` is standalone. “ARC-style” means a compact visual observation, six
integer actions, sparse `levels_completed`, deterministic turns, isolated
preflight, and fresh replay. It does not mean compatibility with or dependence
on an ARC API, engine, loader, metadata format, or game class.

## Actions

| ID | Meaning |
|---:|---|
| 1 | Decrease the selected coordinate by one step |
| 2 | Increase the selected coordinate by one step |
| 3 | Select the previous coordinate |
| 4 | Select the next coordinate |
| 5 | Open the gripper |
| 6 | Close the gripper |

Coordinates cycle `AZIMUTH → REACH → HEIGHT → AZIMUTH`. Step sizes are 5
degrees, 0.020 metres, and 0.015 metres respectively. One control turn is 0.25
seconds. A command can be rejected for reachability or collision; rejection
consumes the action and retains the previous accepted command. The same action
from the same observed state is deterministic in the clean condition.

These action meanings are known apparatus behavior, not a puzzle to decode.

## RGB camera

The scored round returns an exact `numpy.uint8` array with shape
`(72, 128, 3)`. It models decoded and downsampled RGB from a separate C920s:

- deterministic pinhole approximation (unit lens calibration unavailable);
- 78-degree diagonal / 43.3-degree vertical field of view;
- camera position `(0.72, -0.10, 0.50)` metres;
- optical target `(0.14, 0.07, 0.13)` metres;
- world up `(0, 0, 1)`;
- source profile 1920×1080 MJPG at 30 fps with autofocus/auto-light metadata.

The image contains perspective, occlusion, visible surfaces, material color,
directional lighting, contact shadows, and fixed optical vignetting. It contains
no painted HUD, encoded action cells, semantic palette indices, object labels,
success banner, or private-state channel.

## Controller telemetry

`telemetry()` returns separately timestamped controller, arm, and webcam
sections. The host pairs the newest camera frame with a T=1051 reply and records
their skew; it does not claim hardware synchronization. The packet contains:

- selected coordinate, last action, and the T=104 command;
- T=1051 encoder angles, firmware-derived XYZ, signed servo loads,
  torque-enable flags, and voltage;
- C920s source/capture metadata and processed-frame shape;
- host request/response/capture timestamps and generic interlock status.

Camera bytes and telemetry are sealed and hashed independently. A committed
candidate must reproduce both products from its admitted preflight.

Telemetry deliberately omits metric jaw aperture, contact force, collision
reason, object coordinates/identity/attachment, target predicates, support
identity, private contact events, and success logic. Sparse `levels_completed`
and terminal state remain separate host facts.

## World interaction contract

The scored puzzle concerns the initially unknown response of the visible scene,
not the control mapping. Objects persist and collide. Depending on geometry and
contact, they may be pushed, obstructed, enclosed between opposing jaws,
grasped, carried, released, and supported. Unsupported objects fall. Commands
may be rejected by reachability, robot collision, obstacle collision, or a held
object’s swept collision.

No object coordinates, friction values, collision margins, grasp thresholds,
attachment flags, stability formulas, target implementation, level sequence,
or generated seeds are disclosed. Learn those operational relations across
generations from host-sealed RGB frames, paired controller feedback,
sparse `levels_completed`, and bounded isolated preflight scenarios.
Experiments never commit; a candidate can commit only after deterministic host
safety verification and a single-use permit.

The first release is a deterministic simulator experiment. It makes no
sim-to-real claim.
