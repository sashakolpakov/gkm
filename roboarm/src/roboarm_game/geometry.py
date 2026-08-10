"""Small deterministic geometry tests used by swept legality checks."""

from __future__ import annotations

from math import ceil, cos, dist, sin, sqrt

from .kinematics import Vector3
from .world_state import SceneBox


def expanded_contains(box: SceneBox, point: Vector3, margin: float) -> bool:
    return all(
        abs(value - center) <= size * 0.5 + margin
        for value, center, size in zip(point, box.center, box.size)
    )


def segment_intersects_box(
    start: Vector3,
    end: Vector3,
    box: SceneBox,
    radius: float,
) -> bool:
    length = dist(start, end)
    samples = max(2, int(ceil(length / max(radius * 0.5, 0.004))))
    for index in range(samples + 1):
        ratio = index / samples
        point = tuple(
            start_value + (end_value - start_value) * ratio
            for start_value, end_value in zip(start, end)
        )
        if expanded_contains(box, point, radius):
            return True
    return False


def sphere_intersects_box(
    center: Vector3,
    radius: float,
    box: SceneBox,
    margin: float = 0.0,
) -> bool:
    distance_sq = 0.0
    for value, box_center, extent in zip(center, box.center, box.size):
        delta = abs(value - box_center) - extent * 0.5
        if delta > 0.0:
            distance_sq += delta * delta
    return distance_sq < (radius + margin) ** 2


def point_segment_distance(
    point: Vector3,
    start: Vector3,
    end: Vector3,
) -> float:
    segment = tuple(end_value - start_value for start_value, end_value in zip(start, end))
    relative = tuple(value - start_value for value, start_value in zip(point, start))
    length_sq = sum(value * value for value in segment)
    if length_sq <= 1.0e-18:
        return dist(point, start)
    ratio = min(
        1.0,
        max(
            0.0,
            sum(value * direction for value, direction in zip(relative, segment))
            / length_sq,
        ),
    )
    closest = tuple(
        start_value + direction * ratio
        for start_value, direction in zip(start, segment)
    )
    return dist(point, closest)


def segment_segment_distance(
    first_start: Vector3,
    first_end: Vector3,
    second_start: Vector3,
    second_end: Vector3,
) -> float:
    """Return the exact shortest distance between two finite 3-D segments."""

    first = tuple(end - start for start, end in zip(first_start, first_end))
    second = tuple(end - start for start, end in zip(second_start, second_end))
    offset = tuple(start - other for start, other in zip(first_start, second_start))
    a = sum(value * value for value in first)
    b = sum(left * right for left, right in zip(first, second))
    c = sum(value * value for value in second)
    d = sum(left * right for left, right in zip(first, offset))
    e = sum(left * right for left, right in zip(second, offset))
    denominator = a * c - b * b
    epsilon = 1.0e-15

    if a <= epsilon and c <= epsilon:
        return dist(first_start, second_start)
    if a <= epsilon:
        return point_segment_distance(first_start, second_start, second_end)
    if c <= epsilon:
        return point_segment_distance(second_start, first_start, first_end)

    first_numerator: float
    first_denominator = denominator
    second_numerator: float
    second_denominator = denominator
    if denominator < epsilon:
        first_numerator = 0.0
        first_denominator = 1.0
        second_numerator = e
        second_denominator = c
    else:
        first_numerator = b * e - c * d
        second_numerator = a * e - b * d
        if first_numerator < 0.0:
            first_numerator = 0.0
            second_numerator = e
            second_denominator = c
        elif first_numerator > first_denominator:
            first_numerator = first_denominator
            second_numerator = e + b
            second_denominator = c

    if second_numerator < 0.0:
        second_numerator = 0.0
        if -d < 0.0:
            first_numerator = 0.0
        elif -d > a:
            first_numerator = first_denominator
        else:
            first_numerator = -d
            first_denominator = a
    elif second_numerator > second_denominator:
        second_numerator = second_denominator
        if -d + b < 0.0:
            first_numerator = 0.0
        elif -d + b > a:
            first_numerator = first_denominator
        else:
            first_numerator = -d + b
            first_denominator = a

    first_ratio = (
        0.0
        if abs(first_numerator) < epsilon
        else first_numerator / first_denominator
    )
    second_ratio = (
        0.0
        if abs(second_numerator) < epsilon
        else second_numerator / second_denominator
    )
    separation = tuple(
        offset_value + first_ratio * first_value - second_ratio * second_value
        for offset_value, first_value, second_value in zip(offset, first, second)
    )
    return sqrt(sum(value * value for value in separation))


def capsules_overlap(
    first_start: Vector3,
    first_end: Vector3,
    first_radius: float,
    second_start: Vector3,
    second_end: Vector3,
    second_radius: float,
    margin: float = 0.0,
) -> bool:
    return segment_segment_distance(
        first_start,
        first_end,
        second_start,
        second_end,
    ) < first_radius + second_radius + margin


def segment_intersects_vertical_cylinder(
    start: Vector3,
    end: Vector3,
    segment_radius: float,
    *,
    center_x: float,
    center_y: float,
    cylinder_radius: float,
    bottom: float,
    top: float,
) -> bool:
    length = dist(start, end)
    samples = max(
        2,
        int(ceil(length / max(segment_radius * 0.35, 0.003))),
    )
    expanded_radius = cylinder_radius + segment_radius
    for index in range(samples + 1):
        ratio = index / samples
        point = tuple(
            start_value + (end_value - start_value) * ratio
            for start_value, end_value in zip(start, end)
        )
        if not bottom - segment_radius < point[2] < top + segment_radius:
            continue
        radial_sq = (
            (point[0] - center_x) ** 2
            + (point[1] - center_y) ** 2
        )
        if radial_sq < expanded_radius * expanded_radius:
            return True
    return False


def sphere_intersects_vertical_cylinder(
    center: Vector3,
    radius: float,
    *,
    center_x: float,
    center_y: float,
    cylinder_radius: float,
    bottom: float,
    top: float,
) -> bool:
    if center[2] - radius >= top or center[2] + radius <= bottom:
        return False
    radial_distance = sqrt(
        (center[0] - center_x) ** 2 + (center[1] - center_y) ** 2
    )
    return radial_distance < cylinder_radius + radius


def yaw_box_intersects_box(
    center: Vector3,
    size: Vector3,
    yaw: float,
    box: SceneBox,
    margin: float = 0.0,
) -> bool:
    if (
        abs(center[2] - box.center[2]) * 2.0
        >= size[2] + box.size[2] + margin * 2.0
    ):
        return False

    cosine = cos(yaw)
    sine = sin(yaw)
    radial = (cosine, sine)
    tangential = (-sine, cosine)
    delta = (center[0] - box.center[0], center[1] - box.center[1])
    first_half = (size[0] * 0.5, size[1] * 0.5)
    second_half = (box.size[0] * 0.5, box.size[1] * 0.5)
    for axis in ((1.0, 0.0), (0.0, 1.0), radial, tangential):
        center_distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        first_radius = (
            first_half[0] * abs(radial[0] * axis[0] + radial[1] * axis[1])
            + first_half[1]
            * abs(tangential[0] * axis[0] + tangential[1] * axis[1])
        )
        second_radius = (
            second_half[0] * abs(axis[0])
            + second_half[1] * abs(axis[1])
        )
        if center_distance >= first_radius + second_radius + margin:
            return False
    return True


def yaw_box_intersects_vertical_cylinder(
    center: Vector3,
    size: Vector3,
    yaw: float,
    *,
    center_x: float,
    center_y: float,
    cylinder_radius: float,
    bottom: float,
    top: float,
    margin: float = 0.0,
) -> bool:
    if (
        center[2] - size[2] * 0.5 >= top + margin
        or center[2] + size[2] * 0.5 <= bottom - margin
    ):
        return False
    delta_x = center_x - center[0]
    delta_y = center_y - center[1]
    cosine = cos(yaw)
    sine = sin(yaw)
    local_radial = delta_x * cosine + delta_y * sine
    local_tangential = -delta_x * sine + delta_y * cosine
    outside_radial = max(abs(local_radial) - size[0] * 0.5, 0.0)
    outside_tangential = max(abs(local_tangential) - size[1] * 0.5, 0.0)
    expanded_radius = cylinder_radius + margin
    return (
        outside_radial * outside_radial
        + outside_tangential * outside_tangential
        < expanded_radius * expanded_radius
    )


def segment_intersects_yaw_box(
    start: Vector3,
    end: Vector3,
    segment_radius: float,
    center: Vector3,
    size: Vector3,
    yaw: float,
) -> bool:
    length = dist(start, end)
    samples = max(
        2,
        int(ceil(length / max(segment_radius * 0.35, 0.003))),
    )
    cosine = cos(yaw)
    sine = sin(yaw)
    half = tuple(value * 0.5 + segment_radius for value in size)
    for index in range(samples + 1):
        ratio = index / samples
        point = tuple(
            start_value + (end_value - start_value) * ratio
            for start_value, end_value in zip(start, end)
        )
        delta_x = point[0] - center[0]
        delta_y = point[1] - center[1]
        local = (
            delta_x * cosine + delta_y * sine,
            -delta_x * sine + delta_y * cosine,
            point[2] - center[2],
        )
        if all(abs(value) < extent for value, extent in zip(local, half)):
            return True
    return False


def boxes_overlap(
    first_center: Vector3,
    first_size: Vector3,
    second: SceneBox,
    margin: float = 0.0,
) -> bool:
    return all(
        abs(first_value - second_value) * 2.0
        < first_extent + second_extent + 2.0 * margin
        for first_value, first_extent, second_value, second_extent in zip(
            first_center,
            first_size,
            second.center,
            second.size,
        )
    )


def inside_horizontal_target(
    center: Vector3,
    size: Vector3,
    target: SceneBox,
    margin: float = 0.004,
) -> bool:
    return all(
        abs(center[index] - target.center[index]) + size[index] * 0.5
        <= target.size[index] * 0.5 - margin
        for index in (0, 1)
    )


__all__ = [
    "boxes_overlap",
    "capsules_overlap",
    "expanded_contains",
    "inside_horizontal_target",
    "point_segment_distance",
    "segment_intersects_box",
    "segment_intersects_vertical_cylinder",
    "segment_intersects_yaw_box",
    "segment_segment_distance",
    "sphere_intersects_box",
    "sphere_intersects_vertical_cylinder",
    "yaw_box_intersects_box",
    "yaw_box_intersects_vertical_cylinder",
]
