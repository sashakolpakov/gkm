import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import selected_shape, step_many


FIRST_TRANSPORT = (
    [5]
    + [2] + [3] * 4 + [2] * 3 + [1] * 2
    + [1] * 11 + [3] * 9
    + [1] * 2
    + [2] * 3 + [4] * 11
    + [1] * 4
    + [4] * 2 + [2] + [4] * 2 + [2] + [4] * 2
    + [2] * 2 + [3] * 3
    + [2] * 9 + [3] * 4
    + [2] + [3] * 2 + [2] + [3] * 6
)

SECOND_TRANSPORT = (
    [5]
    + [3] * 4 + [2] * 3 + [1] * 2
    + [1] * 9 + [3] * 9
    + [1]
    + [4] * 12 + [1] * 4
    + [4] * 2 + [2] + [4] * 2 + [2] + [4] * 2
    + [2] * 2 + [3] * 3
    + [2] * 9 + [3] * 4
    + [2] * 4 + [3] * 7
)

SECOND_STAGES = (
    ("select", [5]),
    ("horizontal", [3] * 4 + [2] * 3 + [1] * 2),
    ("paint_6", [1] * 9 + [3] * 9),
    ("gap_row", [1]),
    ("gap_right", [4] * 12),
    ("upper_block", [1] * 4),
    ("vertical", [4] * 2 + [2] + [4] * 2 + [2] + [4] * 2),
    ("release", [2] * 2 + [3] * 3),
    ("lower_block", [2] * 9 + [3] * 4),
    ("place", [2] * 4 + [3] * 7),
)


def target_score(node, color, bbox):
    grid = arr(node.frame())
    row0, col0, row1, col1 = bbox
    points = (
        [(row0, col) for col in range(col0, col1 + 1)]
        + [(row1, col) for col in range(col0, col1 + 1)]
        + [(row, col0) for row in range(row0 + 1, row1)]
        + [(row, col1) for row in range(row0 + 1, row1)]
    )
    return sum(int(grid[row, col]) == color for row, col in points), len(points)


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    env = env.clone()
    step_many(env, FIRST_TRANSPORT)
    print(
        "first",
        len(FIRST_TRANSPORT),
        selected_shape(env),
        target_score(env, 11, (39, 9, 57, 15)),
        "level",
        env.levels_completed,
    )
    try:
        index = 0
        for name, path in SECOND_STAGES:
            for action in path:
                index += 1
                env.step(action)
                if env.levels_completed > 7:
                    print("completed", index, "of", len(SECOND_TRANSPORT))
                    return
            print(name, selected_shape(env))
            if name == "gap_right":
                for count in range(1, 9):
                    scout = env.clone()
                    step_many(scout, [4] * count)
                    print("edge_right", count, selected_shape(scout))
    except Exception as error:
        print("second_error", index, type(error).__name__, str(error))
        return
    print(
        "second",
        len(SECOND_TRANSPORT),
        selected_shape(env),
        target_score(env, 6, (45, 6, 54, 21)),
        "first_after",
        target_score(env, 11, (39, 9, 57, 15)),
        "level",
        env.levels_completed,
    )


if __name__ == "__main__":
    A.run_program("re86", probe)
