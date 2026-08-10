"""Deterministic perspective RGB camera for the operational workcell.

This is the actual round observation renderer.  It is intentionally separate
from the browser's Three.js reconstruction: the Gödel–Kolmogorov machine gets
these exact RGB bytes, while the browser merely replays them and visualizes the
same authoritative state at a larger scale.
"""

from __future__ import annotations

from collections import OrderedDict
from math import cos, radians, sin, tan
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .config import (
    BASE_HEIGHT_M,
    BASE_RADIUS_M,
    ELBOW_JOINT_RADIUS_M,
    SHOULDER_JOINT_RADIUS_M,
    TABLE_SIZE_M,
    WORKCELL_REAR_WALL_CENTER_M,
    WORKCELL_REAR_WALL_SIZE_M,
    WRIST_JOINT_RADIUS_M,
)
from .kinematics import Vector3, exact_anchors
from .observation import CAMERA_MODEL, FRAME_SHAPE, RgbFrame
from .physical_geometry import (
    barrier_cap,
    gripper_boxes,
    robot_capsules,
    target_walls,
    workcell_solids,
)
from .world_state import SceneBox, WorldState

FloatArray = NDArray[np.float64]

_HEIGHT: Final[int] = FRAME_SHAPE[0]
_WIDTH: Final[int] = FRAME_SHAPE[1]
_NEAR: Final[float] = float(CAMERA_MODEL["near_m"])
_FAR: Final[float] = float(CAMERA_MODEL["far_m"])
_CAMERA_ORIGIN = np.asarray(CAMERA_MODEL["position_m"], dtype=np.float64)
_CAMERA_TARGET = np.asarray(CAMERA_MODEL["target_m"], dtype=np.float64)
_CAMERA_UP = np.asarray(CAMERA_MODEL["up_axis"], dtype=np.float64)


def _unit(value: FloatArray) -> FloatArray:
    length = float(np.linalg.norm(value))
    if length <= 1.0e-15:
        raise ValueError("cannot normalize a zero vector")
    return value / length


def _camera_rays() -> FloatArray:
    forward = _unit(_CAMERA_TARGET - _CAMERA_ORIGIN)
    right = _unit(np.cross(forward, _CAMERA_UP))
    up = _unit(np.cross(right, forward))
    scale = tan(radians(float(CAMERA_MODEL["vertical_fov_deg"])) * 0.5)
    aspect = _WIDTH / _HEIGHT
    columns = (
        (np.arange(_WIDTH, dtype=np.float64) + 0.5) / _WIDTH * 2.0 - 1.0
    ) * scale * aspect
    rows = (
        1.0
        - (np.arange(_HEIGHT, dtype=np.float64) + 0.5)
        / _HEIGHT
        * 2.0
    ) * scale
    rays = (
        forward[None, None, :]
        + columns[None, :, None] * right[None, None, :]
        + rows[:, None, None] * up[None, None, :]
    )
    return rays / np.linalg.norm(rays, axis=2, keepdims=True)


_RAYS: Final[FloatArray] = _camera_rays()
_IDENTITY_AXES: Final[FloatArray] = np.eye(3, dtype=np.float64)


def _box_hit(
    center: Vector3,
    size: Vector3,
    axes: FloatArray = _IDENTITY_AXES,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Intersect every camera ray with one axis-aligned or oriented box."""

    center_value = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) * 0.5
    local_origin = axes @ (_CAMERA_ORIGIN - center_value)
    local_rays = np.einsum("ij,hwj->hwi", axes, _RAYS)
    safe_rays = np.where(
        np.abs(local_rays) < 1.0e-14,
        np.copysign(1.0e-14, local_rays + 1.0e-30),
        local_rays,
    )
    first = (-half - local_origin) / safe_rays
    second = (half - local_origin) / safe_rays
    near_by_axis = np.minimum(first, second)
    far_by_axis = np.maximum(first, second)
    near = np.max(near_by_axis, axis=2)
    far = np.min(far_by_axis, axis=2)
    distance = np.where(near > _NEAR, near, far)
    valid = (far >= np.maximum(near, _NEAR)) & (distance <= _FAR)

    local_points = local_origin + local_rays * distance[:, :, None]
    normalized = np.abs(local_points / np.maximum(half, 1.0e-12))
    normal_axis = np.argmax(normalized, axis=2)
    local_normals = np.zeros_like(local_points)
    for axis in range(3):
        mask = normal_axis == axis
        local_normals[:, :, axis][mask] = np.where(
            local_points[:, :, axis][mask] >= 0.0,
            1.0,
            -1.0,
        )
    normals = np.einsum("ji,hwj->hwi", axes, local_normals)
    return distance, normals, valid


def _sphere_hit(
    center: Vector3,
    radius: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    center_value = np.asarray(center, dtype=np.float64)
    offset = _CAMERA_ORIGIN - center_value
    half_b = np.einsum("hwi,i->hw", _RAYS, offset)
    c = float(np.dot(offset, offset) - radius * radius)
    discriminant = half_b * half_b - c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = -half_b - root
    far = -half_b + root
    distance = np.where(near > _NEAR, near, far)
    valid = (discriminant >= 0.0) & (distance > _NEAR) & (distance <= _FAR)
    points = _CAMERA_ORIGIN + _RAYS * distance[:, :, None]
    normals = (points - center_value) / radius
    return distance, normals, valid


def _vertical_cylinder_hit(
    center_x: float,
    center_y: float,
    radius: float,
    bottom: float,
    top: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Intersect camera rays with a finite, flat-capped vertical cylinder."""

    offset_x = _CAMERA_ORIGIN[0] - center_x
    offset_y = _CAMERA_ORIGIN[1] - center_y
    ray_x = _RAYS[:, :, 0]
    ray_y = _RAYS[:, :, 1]
    ray_z = _RAYS[:, :, 2]
    quadratic_a = ray_x * ray_x + ray_y * ray_y
    quadratic_b = 2.0 * (offset_x * ray_x + offset_y * ray_y)
    quadratic_c = offset_x * offset_x + offset_y * offset_y - radius * radius
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    safe_a = np.where(np.abs(quadratic_a) < 1.0e-14, 1.0e-14, quadratic_a)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    side_near = (-quadratic_b - root) / (2.0 * safe_a)
    side_far = (-quadratic_b + root) / (2.0 * safe_a)
    side_distance = np.full((_HEIGHT, _WIDTH), _FAR + 1.0, dtype=np.float64)
    for candidate in (side_near, side_far):
        height = _CAMERA_ORIGIN[2] + candidate * ray_z
        candidate_valid = (
            (discriminant >= 0.0)
            & (candidate > _NEAR)
            & (candidate <= _FAR)
            & (height >= bottom)
            & (height <= top)
        )
        side_distance = np.where(
            candidate_valid & (candidate < side_distance),
            candidate,
            side_distance,
        )

    cap_distance = np.full((_HEIGHT, _WIDTH), _FAR + 1.0, dtype=np.float64)
    cap_normal_z = np.zeros((_HEIGHT, _WIDTH), dtype=np.float64)
    safe_ray_z = np.where(
        np.abs(ray_z) < 1.0e-14,
        np.copysign(1.0e-14, ray_z + 1.0e-30),
        ray_z,
    )
    for height, normal_z in ((bottom, -1.0), (top, 1.0)):
        candidate = (height - _CAMERA_ORIGIN[2]) / safe_ray_z
        x = _CAMERA_ORIGIN[0] + candidate * ray_x - center_x
        y = _CAMERA_ORIGIN[1] + candidate * ray_y - center_y
        candidate_valid = (
            (candidate > _NEAR)
            & (candidate <= _FAR)
            & (x * x + y * y <= radius * radius)
        )
        use = candidate_valid & (candidate < cap_distance)
        cap_distance = np.where(use, candidate, cap_distance)
        cap_normal_z = np.where(use, normal_z, cap_normal_z)

    use_side = side_distance <= cap_distance
    distance = np.where(use_side, side_distance, cap_distance)
    valid = distance <= _FAR
    points = _CAMERA_ORIGIN + _RAYS * distance[:, :, None]
    normals = np.zeros_like(points)
    radial_x = points[:, :, 0] - center_x
    radial_y = points[:, :, 1] - center_y
    radial_length = np.maximum(
        np.sqrt(radial_x * radial_x + radial_y * radial_y),
        1.0e-15,
    )
    normals[:, :, 0] = np.where(use_side, radial_x / radial_length, 0.0)
    normals[:, :, 1] = np.where(use_side, radial_y / radial_length, 0.0)
    normals[:, :, 2] = np.where(use_side, 0.0, cap_normal_z)
    return distance, normals, valid


def _capsule_hit(
    start: Vector3,
    end: Vector3,
    radius: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Analytic ray/capsule intersection for one rounded robot segment."""

    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    axis = second - first
    origin = _CAMERA_ORIGIN - first
    axis_length_sq = float(np.dot(axis, axis))
    if axis_length_sq <= 1.0e-15:
        return _sphere_hit(start, radius)

    axis_ray = np.einsum("hwi,i->hw", _RAYS, axis)
    axis_origin = float(np.dot(axis, origin))
    ray_origin = np.einsum("hwi,i->hw", _RAYS, origin)
    origin_sq = float(np.dot(origin, origin))
    quadratic_a = axis_length_sq - axis_ray * axis_ray
    quadratic_b = axis_length_sq * ray_origin - axis_origin * axis_ray
    quadratic_c = (
        axis_length_sq * origin_sq
        - axis_origin * axis_origin
        - radius * radius * axis_length_sq
    )
    discriminant = quadratic_b * quadratic_b - quadratic_a * quadratic_c
    safe_a = np.where(
        np.abs(quadratic_a) < 1.0e-14,
        1.0e-14,
        quadratic_a,
    )
    body_distance = (
        -quadratic_b - np.sqrt(np.maximum(discriminant, 0.0))
    ) / safe_a
    along = axis_origin + body_distance * axis_ray
    body_valid = (
        (discriminant >= 0.0)
        & (body_distance > _NEAR)
        & (body_distance <= _FAR)
        & (along > 0.0)
        & (along < axis_length_sq)
    )

    first_distance, _, first_valid = _sphere_hit(start, radius)
    second_distance, _, second_valid = _sphere_hit(end, radius)
    cap_distance = np.where(
        first_valid & (~second_valid | (first_distance <= second_distance)),
        first_distance,
        second_distance,
    )
    cap_valid = first_valid | second_valid
    use_body = body_valid & (~cap_valid | (body_distance <= cap_distance))
    distance = np.where(use_body, body_distance, cap_distance)
    valid = use_body | cap_valid

    points = _CAMERA_ORIGIN + _RAYS * distance[:, :, None]
    projection = np.clip(
        np.einsum("hwi,i->hw", points - first, axis) / axis_length_sq,
        0.0,
        1.0,
    )
    closest = first + projection[:, :, None] * axis
    normals = (points - closest) / radius
    normals /= np.maximum(
        np.linalg.norm(normals, axis=2, keepdims=True),
        1.0e-15,
    )
    return distance, normals, valid


class _SensorBuffer:
    def __init__(self, template: "_SensorBuffer | None" = None) -> None:
        if template is not None:
            self.color = template.color.copy()
            self.depth = template.depth.copy()
            self.normal = template.normal.copy()
            self.specular = template.specular.copy()
            self.material = template.material.copy()
            return
        row_gradient = np.linspace(0.0, 1.0, _HEIGHT, dtype=np.float64)
        top = np.asarray((24.0, 32.0, 35.0), dtype=np.float64)
        bottom = np.asarray((10.0, 15.0, 18.0), dtype=np.float64)
        self.color = (
            top[None, None, :] * (1.0 - row_gradient[:, None, None])
            + bottom[None, None, :] * row_gradient[:, None, None]
        )
        self.color = np.repeat(self.color, _WIDTH, axis=1)
        self.depth = np.full((_HEIGHT, _WIDTH), _FAR, dtype=np.float64)
        self.normal = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.float64)
        self.specular = np.zeros((_HEIGHT, _WIDTH), dtype=np.float64)
        self.material = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)

    def paint(
        self,
        hit: tuple[FloatArray, FloatArray, FloatArray],
        color: tuple[int, int, int],
        *,
        specular: float = 0.12,
        material: int = 1,
    ) -> None:
        distance, normal, valid = hit
        mask = valid & (distance < self.depth)
        if not np.any(mask):
            return
        self.depth[mask] = distance[mask]
        self.normal[mask] = normal[mask]
        self.color[mask] = np.asarray(color, dtype=np.float64)
        self.specular[mask] = specular
        self.material[mask] = material


def _axes(azimuth: float) -> FloatArray:
    radial = np.asarray((cos(azimuth), sin(azimuth), 0.0), dtype=np.float64)
    tangential = np.asarray((-sin(azimuth), cos(azimuth), 0.0), dtype=np.float64)
    vertical = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    return np.stack((radial, tangential, vertical), axis=0)


def _paint_box(
    buffer: _SensorBuffer,
    box: SceneBox,
    color: tuple[int, int, int],
    *,
    specular: float,
    material: int,
) -> None:
    buffer.paint(
        _box_hit(box.center, box.size),
        color,
        specular=specular,
        material=material,
    )


def _static_scene_geometry(buffer: _SensorBuffer, state: WorldState) -> None:
    # Architectural context establishes perspective, scale, and occlusion.
    buffer.paint(
        _box_hit((0.0, 0.0, -0.105), (2.4, 2.2, 0.045)),
        (35, 42, 44),
        specular=0.03,
        material=2,
    )
    buffer.paint(
        _box_hit(
            (0.0, 0.0, -0.0275),
            (TABLE_SIZE_M[0], TABLE_SIZE_M[1], 0.055),
        ),
        (77, 88, 89),
        specular=0.34,
        material=3,
    )
    buffer.paint(
        _box_hit(
            WORKCELL_REAR_WALL_CENTER_M,
            WORKCELL_REAR_WALL_SIZE_M,
        ),
        (38, 46, 49),
        specular=0.04,
        material=4,
    )
    for index in range(-3, 4):
        shade = (55, 66, 69) if index % 2 == 0 else (47, 57, 60)
        buffer.paint(
            _box_hit((index * 0.19, 0.608, 0.33), (0.016, 0.018, 0.62)),
            shade,
            specular=0.18,
            material=5,
        )
    for post in workcell_solids()[1:]:
        buffer.paint(
            _box_hit(post.center, post.size),
            (104, 118, 121),
            specular=0.55,
            material=5,
        )

    # Obstacle and its visible caution cap.
    _paint_box(
        buffer,
        state.barrier,
        (164, 57, 48),
        specular=0.30,
        material=6,
    )
    cap = barrier_cap(state.barrier)
    buffer.paint(
        _box_hit(cap.center, cap.size),
        (241, 178, 57),
        specular=0.24,
        material=7,
    )

    # Open target bin: floor plus four physical retaining walls.
    _paint_box(
        buffer,
        state.target,
        (26, 151, 143),
        specular=0.28,
        material=8,
    )
    for wall in target_walls(state.target):
        buffer.paint(
            _box_hit(wall.center, wall.size),
            (38, 191, 179),
            specular=0.32,
            material=8,
        )

def _object_geometry(buffer: _SensorBuffer, state: WorldState) -> None:
    # Movable workpiece and its dark orientation mark.
    object_box = SceneBox(
        box_id=state.object.object_id,
        center=state.object.position,
        size=state.object.size,
        color_role=state.object.color_role,
    )
    _paint_box(
        buffer,
        object_box,
        (242, 191, 67),
        specular=0.34,
        material=9,
    )
    buffer.paint(
        _box_hit(
            (
                state.object.position[0],
                state.object.position[1],
                state.object.position[2] + state.object.size[2] * 0.5 + 0.001,
            ),
            (
                state.object.size[0] * 0.72,
                state.object.size[1] * 0.72,
                0.002,
            ),
        ),
        (35, 38, 37),
        specular=0.10,
        material=10,
    )


_STATIC_SCENE_CACHE: OrderedDict[tuple[object, ...], _SensorBuffer] = (
    OrderedDict()
)


def _static_scene_key(state: WorldState) -> tuple[object, ...]:
    return (
        state.barrier.center,
        state.barrier.size,
        state.target.center,
        state.target.size,
    )


def _static_scene(state: WorldState) -> _SensorBuffer:
    key = _static_scene_key(state)
    cached = _STATIC_SCENE_CACHE.get(key)
    if cached is not None:
        _STATIC_SCENE_CACHE.move_to_end(key)
        return cached
    buffer = _SensorBuffer()
    _static_scene_geometry(buffer, state)
    _STATIC_SCENE_CACHE[key] = buffer
    while len(_STATIC_SCENE_CACHE) > 4:
        _STATIC_SCENE_CACHE.popitem(last=False)
    return buffer


def _robot_geometry(buffer: _SensorBuffer, state: WorldState) -> None:
    anchors = exact_anchors(state.robot.joints)
    dark = (35, 50, 56)
    orange = (228, 121, 31)
    rust = (169, 69, 28)
    joint = (31, 42, 47)
    silver = (207, 216, 214)

    buffer.paint(
        _vertical_cylinder_hit(
            0.0,
            0.0,
            BASE_RADIUS_M,
            0.0,
            BASE_HEIGHT_M,
        ),
        dark,
        specular=0.62,
        material=11,
    )
    capsules = robot_capsules(anchors)
    for capsule, color, specular in zip(
        capsules,
        (dark, orange, rust, joint),
        (0.58, 0.42, 0.48, 0.60),
    ):
        buffer.paint(
            _capsule_hit(capsule.start, capsule.end, capsule.radius),
            color,
            specular=specular,
            material=12,
        )
    for center, radius, color, specular in (
        (anchors.shoulder, SHOULDER_JOINT_RADIUS_M, dark, 0.62),
        (anchors.elbow, ELBOW_JOINT_RADIUS_M, joint, 0.66),
        (anchors.wrist, WRIST_JOINT_RADIUS_M, silver, 0.75),
    ):
        buffer.paint(
            _sphere_hit(center, radius),
            color,
            specular=specular,
            material=13,
        )

    azimuth = state.robot.command_azimuth
    basis = _axes(azimuth)
    for body in gripper_boxes(
        anchors.tcp,
        azimuth,
        state.robot.gripper_aperture,
    ):
        buffer.paint(
            _box_hit(
                body.center,
                body.size,
                basis,
            ),
            silver,
            specular=0.78,
            material=14,
        )
    buffer.paint(
        _sphere_hit(anchors.tcp, 0.007),
        (74, 226, 205),
        specular=0.30,
        material=15,
    )


def _table_pattern_and_shadows(
    buffer: _SensorBuffer,
    state: WorldState,
) -> FloatArray:
    multiplier = np.ones((_HEIGHT, _WIDTH), dtype=np.float64)
    table = buffer.material == 3
    if not np.any(table):
        return multiplier
    points = _CAMERA_ORIGIN + _RAYS * buffer.depth[:, :, None]
    x = points[:, :, 0]
    y = points[:, :, 1]
    grid_x = np.abs((x + 0.45) / 0.05 - np.round((x + 0.45) / 0.05))
    grid_y = np.abs((y + 0.45) / 0.05 - np.round((y + 0.45) / 0.05))
    grid = table & ((grid_x < 0.018) | (grid_y < 0.018))
    multiplier[grid] *= 0.70

    anchors = exact_anchors(state.robot.joints)
    casters = [
        (*state.object.position[:2], state.object.position[2], 0.030, 0.30),
        (*state.barrier.center[:2], state.barrier.center[2], 0.060, 0.24),
        (*anchors.shoulder[:2], anchors.shoulder[2], 0.055, 0.20),
        (*anchors.elbow[:2], anchors.elbow[2], 0.050, 0.20),
        (*anchors.tcp[:2], anchors.tcp[2], 0.045, 0.22),
    ]
    for caster_x, caster_y, height, radius, strength in casters:
        shadow_x = caster_x - height * 0.22
        shadow_y = caster_y + height * 0.18
        distance = (
            ((x - shadow_x) / radius) ** 2
            + ((y - shadow_y) / (radius * 0.65)) ** 2
        )
        shadow = 1.0 - strength * np.exp(-distance * 1.8)
        multiplier[table] *= shadow[table]
    return multiplier


def _shade(buffer: _SensorBuffer, state: WorldState) -> RgbFrame:
    hit = buffer.material != 0
    normal = buffer.normal
    light = _unit(np.asarray((0.45, -0.35, 0.82), dtype=np.float64))
    diffuse = np.maximum(0.0, np.einsum("hwi,i->hw", normal, light))
    fill = np.maximum(0.0, normal[:, :, 2])
    brightness = 0.30 + 0.62 * diffuse + 0.08 * fill
    brightness *= _table_pattern_and_shadows(buffer, state)

    view = -_RAYS
    half_vector = light + view
    half_vector /= np.maximum(
        np.linalg.norm(half_vector, axis=2, keepdims=True),
        1.0e-15,
    )
    specular_angle = np.maximum(
        0.0,
        np.sum(normal * half_vector, axis=2),
    )
    highlight = (
        np.power(specular_angle, 28.0)
        * buffer.specular
        * 150.0
    )

    linear = np.power(np.clip(buffer.color / 255.0, 0.0, 1.0), 2.2)
    lit = linear * brightness[:, :, None]
    lit += (highlight / 255.0)[:, :, None]
    rgb = np.power(np.clip(lit, 0.0, 1.0), 1.0 / 2.2) * 255.0
    rgb[~hit] = buffer.color[~hit]

    # A fixed optical vignette is part of the camera model, not random noise.
    x = (
        (np.arange(_WIDTH, dtype=np.float64) + 0.5) / _WIDTH * 2.0 - 1.0
    )
    y = (
        (np.arange(_HEIGHT, dtype=np.float64) + 0.5) / _HEIGHT * 2.0 - 1.0
    )
    radius_sq = y[:, None] ** 2 + x[None, :] ** 2
    vignette = np.clip(1.0 - 0.12 * radius_sq, 0.78, 1.0)
    rgb *= vignette[:, :, None]
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


_FRAME_CACHE: OrderedDict[tuple[object, ...], RgbFrame] = OrderedDict()
_FRAME_CACHE_LIMIT: Final[int] = 1024


def _frame_key(state: WorldState) -> tuple[object, ...]:
    return (
        *_static_scene_key(state),
        state.robot.joints,
        state.robot.command_azimuth,
        state.robot.gripper_aperture,
        state.object.position,
        state.success,
    )


def render_operational(state: WorldState) -> RgbFrame:
    """Render one exact 128×72×3 RGB8 C920s-approximation observation."""

    key = _frame_key(state)
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        _FRAME_CACHE.move_to_end(key)
        return cached.copy()

    buffer = _SensorBuffer(_static_scene(state))
    _object_geometry(buffer, state)
    _robot_geometry(buffer, state)
    frame = _shade(buffer, state)
    if frame.shape != FRAME_SHAPE or frame.dtype != np.uint8:
        raise AssertionError("operational RGB renderer violated its contract")
    frame.setflags(write=False)
    _FRAME_CACHE[key] = frame
    while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
        _FRAME_CACHE.popitem(last=False)
    return frame.copy()


__all__ = ["render_operational"]
