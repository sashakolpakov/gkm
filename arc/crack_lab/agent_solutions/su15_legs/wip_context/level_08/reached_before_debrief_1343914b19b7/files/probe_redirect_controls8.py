import numpy as np

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
    (6, 37, 49),
    (6, 8, 19),
    (6, 8, 31),
    (6, 10, 42),
    (6, 20, 43),
    (6, 21, 43),
    (6, 30, 50),
    (6, 39, 49),
    (6, 40, 51),
)


def inspect(env):
    H.resumed_solve(env)
    start_level = int(env.levels_completed)
    initial = np.asarray(env.frame())
    ring_masks = tuple(
        frozenset(
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row, col]) == 9
        )
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )

    def points(node, color):
        frame = np.asarray(node.frame())
        rows, cols = np.where(frame[10:] == color)
        return frozenset(
            (int(row + 10), int(col)) for row, col in zip(rows, cols)
        )

    def center(value):
        return (
            round(sum(row for row, _ in value) / len(value)),
            round(sum(col for _, col in value) / len(value)),
        )

    def state(node):
        star = points(node, 14)
        cutter = points(node, 7)
        squares = points(node, 12)
        if len(star) != 8 or len(cutter) != 8 or len(squares) != 50:
            return None
        occupancy = tuple(
            max(
                len(item & mask)
                for item in (star, cutter, squares)
            )
            for mask in ring_masks
        )
        return center(star), center(cutter), occupancy, star, cutter

    node = env.clone()
    for action in PREFIX:
        node.step(*action)
    print("ROOT", state(node))
    for depth in range(1, 25):
        current = state(node)
        outcomes = []
        for row, col in current[3] | current[4]:
            if row >= 63 or col <= 0 or col >= 63:
                continue
            action = (6, col, row)
            child = node.clone()
            child.step(*action)
            value = state(child)
            if value is None or value == current:
                continue
            if int(child.levels_completed) > start_level:
                print("FOUND", PREFIX + tuple(item[1] for item in outcomes) + (action,))
                return
            clearance = max(
                abs(value[0][0] - value[1][0]),
                abs(value[0][1] - value[1][1]),
            )
            score = (
                max(value[2][1], value[2][3]),
                value[0][1] - max(0, 8 - clearance) * 4,
                clearance,
            )
            outcomes.append((score, action, child, value))
        if not outcomes:
            print("STUCK", depth)
            return
        best = max(outcomes, key=lambda item: item[0])
        print("STEP", depth, best[1], best[3][:3], "score", best[0])
        node = best[2]
    print("NO_WIN")


H.A.run_program("su15", inspect)
