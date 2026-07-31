import itertools
import json

import gkm_try as H

from perception import connected_components


PREFIX = (
    (6, 7, 48),
    (6, 7, 54), (6, 7, 56), (6, 7, 54), (6, 7, 56),
    (6, 12, 50), (6, 12, 44),
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
            point = todo.pop()
            group.append(point)
            near = {
                other for other in points
                if max(
                    abs(point[0] - other[0]), abs(point[1] - other[1])
                ) <= 1
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
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
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
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def targets(node):
        frame = node.frame()
        squares = tuple(
            frozenset(
                (row, col)
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
                for col in range(blob.bbox[1], blob.bbox[3] + 1)
            )
            for blob in connected_components(
                frame, colors=(12,), min_area=25
            )
            if (
                blob.bbox[0] >= 10
                and blob.size == (5, 5)
                and blob.area == 25
            )
        )
        stars = tuple(frozenset(group) for group in groups(frame, 14))
        return squares + stars

    def summary(node):
        items = targets(node)
        movers = groups(node.frame(), 7)
        overlaps = tuple(
            max((len(item & mask) for mask in ring_masks), default=0)
            for item in items
        )
        centers = tuple(center(item) for item in items)
        return centers, overlaps, tuple(center(group) for group in movers)

    def distance(node):
        items = targets(node)
        if len(items) != 3:
            return 999
        item_centers = tuple(center(item) for item in items)
        return min(
            sum(
                max(abs(row - target_row), abs(col - target_col))
                for (row, col), (target_row, target_col)
                in zip(item_centers, assignment)
            )
            for assignment in itertools.permutations(ring_centers, 3)
        )

    print("ROOT", summary(root), "distance", distance(root))
    frame = root.frame()
    items = targets(root)
    proposed = {(6, 0, 0), (6, 32, 32)}
    for item in items:
        row, col = center(item)
        proposed.add((6, col, row))
        for target_row, target_col in ring_centers:
            proposed.add((
                6,
                col + max(-6, min(6, target_col - col)),
                row + max(-6, min(6, target_row - row)),
            ))
        proposed.update((6, point_col, point_row)
                        for point_row, point_col in item)
    for color in (7, 14):
        proposed.update(
            (6, col, row)
            for group in groups(frame, color)
            for row, col in group
        )
    proposed.update((6, col, row) for row, col in ring_centers)

    outcomes = {}
    for action in sorted(proposed):
        child = root.clone()
        child.step(*action)
        key = (
            int(child.levels_completed), child.terminal(),
            distance(child), summary(child),
        )
        outcomes.setdefault(key, []).append(action)
    print("ACTIONS", len(proposed), "OUTCOMES", len(outcomes))
    for outcome, actions in sorted(
        outcomes.items(), key=lambda item: (-item[0][0], item[0][2])
    )[:24]:
        print("OUT", outcome, "via", actions[:4])
    for color in (14, 7):
        for group in groups(frame, color):
            base_row, base_col = center(group)
            for row, col in group:
                child = root.clone()
                child.step(6, col, row)
                print(
                    "CONTROL", color, (row - base_row, col - base_col),
                    summary(child), "distance", distance(child),
                )
    print("PREFIX", PREFIX, "level", start_level)


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
