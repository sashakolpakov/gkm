import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import SEED, selected_shape, step_many


def trace(node, path):
    previous = selected_shape(node)
    print(0, previous)
    for index, action in enumerate(path, 1):
        node.step(action)
        current = selected_shape(node)
        if (
            current is None
            or previous is None
            or current[1:] != previous[1:]
        ):
            print(index, action, current)
        previous = current
    print("final", selected_shape(node), node.levels_completed)


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, SEED)
    lock_tests = {
        "direct_down": [2] * 10,
        "cycle_down": [5, 5] + [2] * 10,
        "cycle_move_cycle_down": [5, 4, 3, 5] + [2] * 10,
        "reverse_seed": [
            {1: 2, 2: 1, 3: 4, 4: 3}[action]
            for action in reversed(SEED)
        ],
    }
    for name, path in lock_tests.items():
        node = env.clone()
        step_many(node, path)
        print(name, selected_shape(node), node.levels_completed)
    env.step(5)
    tests = {
        "left14_up9_down6": [3] * 14 + [1] * 9 + [2] * 6,
        "left13_up9_down6": [3] * 13 + [1] * 9 + [2] * 6,
        "left15_up9_down6": [3] * 15 + [1] * 9 + [2] * 6,
        "up9_left14_down6": [1] * 9 + [3] * 14 + [2] * 6,
    }
    for name, path in tests.items():
        print(name)
        trace(env.clone(), path)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
