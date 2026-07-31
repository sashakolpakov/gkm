import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups


STAGE = (
    (6, 7, 48),
    (6, 7, 54),
    (6, 15, 19),
    (6, 7, 19),
    (6, 16, 53),
    (6, 31, 44),
    (6, 42, 43),
    (6, 10, 53),
    (6, 7, 55),
)
SET_LEFT = ((6, 37, 49), (6, 8, 19))


def inspect(env):
    H.resumed_solve(env)
    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_masks = tuple(
        frozenset(
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        )
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    ring_centers = tuple(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )

    def state(node):
        frame = node.frame()
        squares = tuple(
            (
                blob.color,
                (round(blob.centroid[0]), round(blob.centroid[1])),
                frozenset(
                    (row, col)
                    for row in range(blob.bbox[0], blob.bbox[2] + 1)
                    for col in range(blob.bbox[1], blob.bbox[3] + 1)
                ),
            )
            for blob in connected_components(
                frame, colors=(12,), min_area=25
            )
            if blob.size == (5, 5) and blob.area == 25
        )
        agents = tuple(
            (color, center(group), frozenset(group))
            for color in (7, 14)
            for group in groups(frame, color)
        )
        items = tuple(points for _, _, points in squares + agents)
        occupancy = tuple(
            max((len(points & mask) for points in items), default=0)
            for mask in ring_masks
        )
        return (
            tuple((color, point) for color, point, _ in squares),
            tuple((color, point) for color, point, _ in agents),
            occupancy,
        )

    node = env.clone()
    for action in STAGE:
        node.step(*action)
    print("STAGED", state(node))
    for action in SET_LEFT:
        node.step(*action)
        print("SET", action, state(node))

    frame = node.frame()
    actions = {
        (6, col, row)
        for color in (7, 14)
        for group in groups(frame, color)
        for row, col in group
    }
    actions.update((6, col, row) for row, col in ring_centers)
    outcomes = {}
    for action in sorted(actions):
        child = node.clone()
        child.step(*action)
        value = (
            int(child.levels_completed),
            bool(child.terminal()),
            state(child),
        )
        outcomes.setdefault(value, []).append(action)
    print("OUTCOMES", len(outcomes))
    for value, actions_for_value in sorted(
        outcomes.items(),
        key=lambda item: (
            -item[0][0],
            -sum(value == 25 for value in item[0][2][2]),
            -sum(item[0][2][2]),
        ),
    ):
        print("OUT", value, "VIA", actions_for_value)
        if value[0] > start_level:
            break


H.A.run_program("su15", inspect)
