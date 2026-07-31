import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


STAGE = [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4


def movable(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(11, 14), min_area=2
        )
    )


def run(root, suffix):
    node = root.clone()
    path = STAGE + suffix
    previous = movable(node)
    trace = []
    for index, action in enumerate(path, 1):
        node.step(action)
        current = movable(node)
        if current != previous and index >= len(STAGE):
            trace.append((index - len(STAGE), action, current))
        previous = current
    return node.levels_completed, trace


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    for name, suffix in (
        ("straight_up", [1] * 16),
        ("left_up", [3] * 2 + [1] * 16),
        ("right_up", [4] * 2 + [1] * 16),
    ):
        print(name, run(env, suffix))
    node = env.clone()
    for action in [3] * 7:
        node.step(action)
    node.step(6, 35, 52)
    previous = movable(node)
    print("reverse_base", previous)
    for index in range(1, 9):
        node.step(1)
        current = movable(node)
        if current != previous:
            print("reverse_up", index, current)
        previous = current
    for name, suffix in (
        ("agent_right", [4] * 12),
        ("agent_up_right", [4] * 4 + [1] * 6),
        ("agent_down_right", [2] * 2 + [4] * 6 + [1] * 8),
    ):
        child = env.clone()
        for action in [3] * 7:
            child.step(action)
        child.step(6, 35, 52)
        for action in [1] * 2 + suffix:
            child.step(action)
        print(name, movable(child))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
