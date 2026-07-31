import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def state(env):
    pieces = [
        (b.color, b.bbox)
        for b in connected_components(
            env.frame(), colors=(6, 10, 11, 15), min_area=1
        )
        if b.bbox[0] >= 10
    ]
    frame = np.asarray(env.frame())
    points = set(map(tuple, np.argwhere(frame == 7)))
    groups = []
    while points:
        todo = [points.pop()]
        group = []
        while todo:
            point = todo.pop()
            group.append(point)
            near = {
                other for other in points
                if max(abs(point[0] - other[0]), abs(point[1] - other[1])) <= 1
            }
            points -= near
            todo.extend(near)
        if max(row for row, _ in group) >= 10:
            groups.append(
                (
                    round(sum(row for row, _ in group) / len(group)),
                    round(sum(col for _, col in group) / len(group)),
                )
            )
    return tuple(pieces), tuple(sorted(groups))


def inspect(env):
    solver.solve(env)
    for name, path in (
        ("left_right", [(6, 7, 26), (6, 48, 27)]),
        ("right_left", [(6, 48, 27), (6, 7, 26)]),
        ("hold_left", [(6, 6, 39), (6, 7, 26), (6, 48, 27)]),
        ("hold_right", [(6, 48, 39), (6, 48, 27), (6, 7, 26)]),
    ):
        clone = env.clone()
        print(name, 0, state(clone))
        for index, action in enumerate(path, 1):
            try:
                clone.step(*action)
                print(name, index, state(clone), "level", clone.levels_completed)
            except Exception as exc:
                print(name, index, type(exc).__name__)
                break


A.run_program("su15", inspect)
