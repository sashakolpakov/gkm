import gkm_try as H

from perception import connected_components


PREFIX = (
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


def groups(frame, color):
    points = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(frame[row][col]) == color
    }
    found = []
    while points:
        todo = [points.pop()]
        group = []
        while todo:
            row, col = todo.pop()
            group.append((row, col))
            near = {
                point for point in points
                if max(abs(point[0] - row), abs(point[1] - col)) <= 1
            }
            points -= near
            todo.extend(near)
        if len(group) >= 4:
            found.append(tuple(sorted(group)))
    return tuple(sorted(found))


def center(points):
    return (
        round(sum(row for row, _ in points) / len(points)),
        round(sum(col for _, col in points) / len(points)),
    )


def inspect(env):
    H.resumed_solve(env)
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

    def summary(node):
        frame = node.frame()
        solids = tuple(
            (
                blob.color,
                (
                    round(blob.centroid[0]),
                    round(blob.centroid[1]),
                ),
                blob.size,
            )
            for blob in connected_components(frame, min_area=4)
            if (
                blob.bbox[0] >= 10
                and blob.area == blob.size[0] * blob.size[1]
                and blob.color not in (3, 4, 5, 9)
            )
        )
        agents = tuple(
            (color, center(group), len(group))
            for color in (7, 14)
            for group in groups(frame, color)
        )
        items = tuple(
            frozenset(
                (row, col)
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
                for col in range(blob.bbox[1], blob.bbox[3] + 1)
            )
            for blob in connected_components(frame, min_area=4)
            if (
                blob.bbox[0] >= 10
                and blob.color in (8, 11, 12)
                and blob.area == blob.size[0] * blob.size[1]
            )
        ) + tuple(
            frozenset(group)
            for color in (7, 14)
            for group in groups(frame, color)
        )
        overlap = tuple(
            max((len(item & mask) for mask in ring_masks), default=0)
            for item in items
        )
        return solids, agents, overlap

    print(
        "ROOT",
        "level", int(env.levels_completed),
        "actions", env.actions,
        summary(env),
    )
    node = env.clone()
    for index, action in enumerate(PREFIX, 1):
        node.step(*action)
        print(index, action, "level", int(node.levels_completed), summary(node))


H.A.run_program("su15", inspect)
