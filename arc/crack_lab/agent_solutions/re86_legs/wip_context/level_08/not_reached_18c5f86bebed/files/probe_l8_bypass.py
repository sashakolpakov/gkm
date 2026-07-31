import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


TARGET = (
    [(39, col) for col in range(9, 16)]
    + [(57, col) for col in range(9, 16)]
    + [(row, 9) for row in range(40, 57)]
    + [(row, 15) for row in range(40, 57)]
)


def score(node):
    grid = arr(node.frame())
    correct = sum(int(grid[row, col]) == 11 for row, col in TARGET)
    return correct, selected_shape(node), node.levels_completed


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, SEED)
    root = env.clone()
    print("root", score(root))

    best = (-1, None, None)
    families = (
        ("top_left", 3, 4, range(5, 14)),
        ("top_right", 4, 3, range(18, 25)),
    )
    for name, outward, inward, side_counts in families:
        for up_count in range(4, 11):
            for side_count in side_counts:
                path = (
                    [1] * up_count
                    + [outward] * side_count
                    + [2] * (up_count + 10)
                    + [inward] * side_count
                )
                node = root.clone()
                step_many(node, path)
                result = score(node)
                if result[0] > best[0]:
                    best = (result[0], name, path, result)
                if result[0] == len(TARGET):
                    print("FOUND", name, path, result)
                    return
    print("BEST", best)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
