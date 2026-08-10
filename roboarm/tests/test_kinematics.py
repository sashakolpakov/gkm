from __future__ import annotations

from math import pi

import numpy as np

from roboarm_game.config import (
    BASE_AXIS_X_OFFSET_M,
    KINEMATIC_TOLERANCE_M,
    SHOULDER_HEIGHT_M,
)
from roboarm_game.kinematics import (
    cylindrical_from_tcp,
    exact_anchors,
    exact_transforms,
    interpolate_joints,
    solve_cylindrical,
)


def test_zero_pose_exact_chain_matches_preserved_xacro_geometry() -> None:
    anchors = exact_anchors((0.0, 0.0, 0.0))
    assert np.allclose(
        anchors.shoulder,
        (BASE_AXIS_X_OFFSET_M, 0.0, SHOULDER_HEIGHT_M),
        atol=1e-12,
    )
    assert np.allclose(
        anchors.elbow,
        (0.040002471103605, -1.102271092340012e-07, 0.359874293621211),
        atol=2e-6,
    )
    assert np.allclose(
        anchors.tcp,
        (0.042001530522195, -1.175548245092273e-07, 0.640074293618902),
        atol=2e-6,
    )
    transforms = exact_transforms((0.0, 0.0, 0.0))
    assert set(transforms) == {
        "world",
        "base_link",
        "link1",
        "link2",
        "link3",
        "gripper_link",
        "hand_tcp",
    }


def test_fixed_branch_ik_hits_canonical_command_lattice() -> None:
    for azimuth in (-40.0 * pi / 180.0, 0.0, 40.0 * pi / 180.0):
        for reach in (0.24, 0.30, 0.38):
            for height in (0.045, 0.15, 0.27, 0.36):
                result = solve_cylindrical(azimuth, reach, height)
                assert result is not None
                observed = cylindrical_from_tcp(result.tcp)
                assert abs(observed[0] - azimuth) < 1e-8
                assert abs(observed[1] - reach) < KINEMATIC_TOLERANCE_M
                assert abs(observed[2] - height) < KINEMATIC_TOLERANCE_M
                assert result.error_m < KINEMATIC_TOLERANCE_M


def test_unreachable_target_is_rejected() -> None:
    assert solve_cylindrical(0.0, 0.8, 0.8) is None
    assert solve_cylindrical(0.0, 0.01, 0.123) is None


def test_joint_interpolation_is_bounded_and_endpoint_exact() -> None:
    start = (0.0, 0.0, 0.0)
    end = (0.4, -0.3, 0.2)
    path = interpolate_joints(start, end, max_delta=0.025)
    previous = start
    for current in path:
        assert max(abs(a - b) for a, b in zip(previous, current)) <= 0.025 + 1e-12
        previous = current
    assert np.allclose(path[-1], end, atol=1e-15)
