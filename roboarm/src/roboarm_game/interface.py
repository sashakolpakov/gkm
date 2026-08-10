"""Public, documented actuator and sensor contract for rb01.

The six action IDs are ordinary integers defined here.  They are not aliases
for actions from an external engine.
"""

from __future__ import annotations

from enum import IntEnum
from types import MappingProxyType
from typing import Final


class Action(IntEnum):
    """Documented coordinate-controller actions."""

    DECREASE = 1
    INCREASE = 2
    PREVIOUS_COORDINATE = 3
    NEXT_COORDINATE = 4
    OPEN_GRIPPER = 5
    CLOSE_GRIPPER = 6


class Coordinate(IntEnum):
    """Cyclic order of the disclosed command coordinates."""

    AZIMUTH = 0
    REACH = 1
    HEIGHT = 2


ACTIONS: Final[tuple[int, ...]] = tuple(int(action) for action in Action)
COORDINATES: Final[tuple[Coordinate, ...]] = tuple(Coordinate)

AZIMUTH_STEP_DEG: Final[float] = 5.0
REACH_STEP_M: Final[float] = 0.020
HEIGHT_STEP_M: Final[float] = 0.015
CONTROL_TURN_S: Final[float] = 0.25

ACTION_MEANINGS = MappingProxyType(
    {
        int(Action.DECREASE): "decrease selected coordinate by one step",
        int(Action.INCREASE): "increase selected coordinate by one step",
        int(Action.PREVIOUS_COORDINATE): "select previous coordinate",
        int(Action.NEXT_COORDINATE): "select next coordinate",
        int(Action.OPEN_GRIPPER): "open gripper",
        int(Action.CLOSE_GRIPPER): "close gripper",
    }
)

COORDINATE_STEPS = MappingProxyType(
    {
        Coordinate.AZIMUTH: AZIMUTH_STEP_DEG,
        Coordinate.REACH: REACH_STEP_M,
        Coordinate.HEIGHT: HEIGHT_STEP_M,
    }
)

KNOWN_INTERFACE: Final[str] = f"""\
rb01-v1 uses integer actions {ACTIONS}.
1/2 decrease/increase the selected coordinate.
3/4 select the previous/next coordinate in the cyclic order
AZIMUTH -> REACH -> HEIGHT -> AZIMUTH.
5 opens and 6 closes the gripper.
Steps: azimuth={AZIMUTH_STEP_DEG} degrees, reach={REACH_STEP_M} metres,
height={HEIGHT_STEP_M} metres; one control turn={CONTROL_TURN_S} seconds.
Commands may be rejected for reachability or collision.  A rejection consumes
the action but retains the previous accepted command.

The scored round returns one deterministic 128×72×3 uint8 RGB frame modeling a
downsampled Logitech C920s capture plus separately timestamped RoArm feedback.
The packet mirrors stock T=1051 encoder/derived-XYZ/load/torque-switch/voltage
fields, identifies the 1080p/30 UVC source, and records host pairing skew.
Selected coordinate, last action, T=104 command, and interlock status are
explicit controller state, not physical sensors. No metric aperture, contact
force, collision reason, or object state is exposed. Telemetry is not painted
into camera pixels."""

KNOWN_WORLD_INTERACTION: Final[str] = """\
Objects persist and collide. Depending on visible geometry and contact, they
may obstruct motion, be enclosed between opposing jaws, grasped, carried,
released, and supported. Released objects settle downward onto the highest
valid support. The useful contact geometry, grasp conditions, clearance,
release behavior, and exact success relation are not supplied and must be
learned from public RGB frames, paired controller feedback, sparse
levels_completed, and bounded host-run preflights.
"""

__all__ = [
    "ACTION_MEANINGS",
    "ACTIONS",
    "AZIMUTH_STEP_DEG",
    "CONTROL_TURN_S",
    "COORDINATE_STEPS",
    "COORDINATES",
    "HEIGHT_STEP_M",
    "KNOWN_INTERFACE",
    "KNOWN_WORLD_INTERACTION",
    "REACH_STEP_M",
    "Action",
    "Coordinate",
]
