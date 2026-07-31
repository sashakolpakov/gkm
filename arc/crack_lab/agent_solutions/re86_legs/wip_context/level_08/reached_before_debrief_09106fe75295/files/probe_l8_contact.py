import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import selected_shape, step_many


FIRST = [3] * 8 + [2] + [4] * 2 + [1] + [3] * 6
RINGS_11 = ((39, 9), (39, 15), (57, 9), (57, 15))


def summary(node):
    grid = arr(node.frame())
    return (
        selected_shape(node),
        tuple(int(grid[row, col]) for row, col in RINGS_11),
        int((grid == 11).sum()),
        node.levels_completed,
    )


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, FIRST)
    print("root", summary(env))
    tests = {
        "up_down": [1, 2],
        "down_up": [2, 1],
        "left_right": [3, 4],
        "right_left": [4, 3],
        "vertical_rub": [1, 2] * 4,
        "horizontal_rub": [3, 4] * 4,
        "box": [1, 4, 2, 3] * 3,
    }
    for name, path in tests.items():
        node = env.clone()
        step_many(node, path)
        print(name, path, summary(node))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
