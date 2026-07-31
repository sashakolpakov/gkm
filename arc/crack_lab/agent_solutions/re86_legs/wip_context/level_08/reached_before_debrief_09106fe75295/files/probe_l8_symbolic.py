import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import selected_shape, step_many


FIRST = [3] * 8 + [2] + [4] * 2 + [1] + [3] * 6


def compact(shape):
    if shape is None:
        return None
    center, color, area, bbox = shape
    return center, color, area, (
        bbox[2] - bbox[0] + 1,
        bbox[3] - bbox[1] + 1,
    ), bbox


def trace(node, path):
    previous = compact(selected_shape(node))
    print(0, previous)
    for index, action in enumerate(path, 1):
        node.step(action)
        current = compact(selected_shape(node))
        if (
            current is None
            or previous is None
            or current[1:] != previous[1:]
        ):
            print(index, action, current)
        previous = current
    print("final", len(path), previous, "level", node.levels_completed)


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    print("first_trace")
    trace(env.clone(), FIRST)

    step_many(env, FIRST)
    env.step(5)
    tests = {
        "up_at_col54": [1] * 16,
        "left_to_col30_then_up": [3] * 8 + [1] * 16,
        "left_to_col21_then_up": [3] * 11 + [1] * 16,
        "left_to_col12_then_up": [3] * 14 + [1] * 16,
        "up_then_left": [1] * 16 + [3] * 18,
    }
    for name, path in tests.items():
        print(name)
        trace(env.clone(), path)

    completion = [2] + [3] * 4 + [2] + [3] * 9
    print("completion")
    trace(env.clone(), completion)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
