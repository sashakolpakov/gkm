import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import selected_shape, step_many


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    initial = env.clone()
    for action, name in ((2, "down"), (4, "right"), (1, "up"), (3, "left")):
        node = initial.clone()
        for count in range(1, 9):
            node.step(action)
            print("initial", name, count, selected_shape(node))

    horizontal = initial.clone()
    horizontal.step(5)
    step_many(horizontal, [2] + [3] * 4)
    print("horizontal_root", selected_shape(horizontal))
    for count in range(1, 6):
        node = horizontal.clone()
        step_many(node, [2] * count)
        print("horizontal_down", count, selected_shape(node))

    step_many(env, [3] * 8 + [2])
    print("root", selected_shape(env))
    for count in range(1, 7):
        node = env.clone()
        step_many(node, [4] * count)
        print("right", count, selected_shape(node))
    pinned = env.clone()
    step_many(pinned, [4] * 3)
    around = pinned.clone()
    step_many(around, [1] * 2 + [4] * 3 + [2] * 2)
    print("around_block", selected_shape(around))
    for pushes in range(1, 7):
        node = around.clone()
        step_many(node, [3] * pushes)
        print("around_left", pushes, selected_shape(node))
    for vertical, name in ((1, "up"), (2, "down")):
        for shift in range(1, 9):
            for pushes in range(1, 4):
                node = pinned.clone()
                step_many(node, [vertical] * shift + [4] * pushes)
                print(
                    "pinned",
                    name,
                    shift,
                    "right",
                    pushes,
                    selected_shape(node),
                )

    for rights in range(2, 6):
        node = env.clone()
        route = [4] * rights + [3] + [1] * 9 + [3] * 3 + [1] + [3] * 2 + [1]
        step_many(node, route)
        print("paint_route", rights, selected_shape(node))
        if rights == 3:
            step_many(node, [1] * 3 + [3] * 4)
            print("narrow_painted_at_edge", selected_shape(node))
            for down in range(1, 16):
                node.step(2)
                print("edge_down", down, selected_shape(node))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
