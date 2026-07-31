import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


TARGET_11 = (
    [(39, col) for col in range(9, 16)]
    + [(57, col) for col in range(9, 16)]
    + [(row, 9) for row in range(40, 57)]
    + [(row, 15) for row in range(40, 57)]
)


def summary(node):
    grid = arr(node.frame())
    return (
        selected_shape(node),
        sum(int(grid[row, col]) == 11 for row, col in TARGET_11),
        int((grid == 11).sum()),
        node.levels_completed,
    )


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, SEED)
    print("painted", summary(env))
    tests = {
        "right1": [4],
        "right2": [4] * 2,
        **{f"right{count}": [4] * count for count in range(3, 13)},
        "left1": [3],
        "left2": [3] * 2,
        "left3_only": [3] * 3,
        "left4_only": [3] * 4,
        "up1": [1],
        "up4": [1] * 4,
        "down1": [2],
        "down4": [2] * 4,
        "left3_down10_right3": [3] * 3 + [2] * 10 + [4] * 3,
        "left4_down10_right4": [3] * 4 + [2] * 10 + [4] * 4,
        "left5_down10_right5": [3] * 5 + [2] * 10 + [4] * 5,
        "left8_down10_right8": [3] * 8 + [2] * 10 + [4] * 8,
        "left10_down10_right10": [3] * 10 + [2] * 10 + [4] * 10,
        "right20_down10_left20": [4] * 20 + [2] * 10 + [3] * 20,
    }
    for name, path in tests.items():
        node = env.clone()
        step_many(node, path)
        print(name, summary(node))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
