import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import selected_shape, step_many


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    env.step(5)
    stages = (
        ("horizontal", [2] + [3] * 4 + [2] * 3 + [1] * 2),
        ("gap_left", [1] * 11 + [3] * 9),
        ("paint_11", [1] * 2),
        ("gap_right", [2] * 3 + [4] * 11),
        ("upper_block", [1] * 4),
    )
    for name, path in stages:
        step_many(env, path)
        print(name, selected_shape(env))

    for count in range(1, 7):
        node = env.clone()
        step_many(node, [4] * count)
        print("upper_right", count, selected_shape(node))
    tests = {
        "r2_d_r4": [4] * 2 + [2] + [4] * 4,
        "r3_d_r3": [4] * 3 + [2] + [4] * 3,
        "r2_d_r2_d_r2": [4] * 2 + [2] + [4] * 2 + [2] + [4] * 2,
        "r2_d_r2_d_r3": [4] * 2 + [2] + [4] * 2 + [2] + [4] * 3,
        "r3_d_r2_d_r": [4] * 3 + [2] + [4] * 2 + [2, 4],
    }
    for name, path in tests.items():
        node = env.clone()
        step_many(node, path)
        print(name, selected_shape(node))
        if name == "r2_d_r2_d_r2":
            step_many(node, [2] * 2 + [3] * 3)
            print("right_edge", selected_shape(node))
            for down in range(1, 14):
                node.step(2)
                print("right_edge_down", down, selected_shape(node))
                if down == 10:
                    for left in range(1, 9):
                        placed = node.clone()
                        step_many(placed, [3] * left)
                        print("lower_left", left, selected_shape(placed))
                    pinned_lower = node.clone()
                    step_many(pinned_lower, [3] * 2)
                    for right in range(1, 8):
                        expanded = pinned_lower.clone()
                        step_many(expanded, [4] * right)
                        print("lower_expand", right, selected_shape(expanded))
                    above = node.clone()
                    step_many(above, [1] + [3] * 4)
                    print("above_lower", selected_shape(above))
                    for pushes in range(1, 8):
                        expanded = above.clone()
                        step_many(expanded, [2] * pushes)
                        print("lower_down_expand", pushes, selected_shape(expanded))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
