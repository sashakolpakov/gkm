import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def pieces(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(0, 5, 11, 12, 13, 14), min_area=2
        )
    )


def run(env, click, path):
    node = env.clone()
    if click:
        node.step(6, *click)
    trace = [(0, pieces(node))]
    for i, action in enumerate(path, 1):
        node.step(*action) if isinstance(action, tuple) else node.step(action)
        if i == len(path) or i % 3 == 0:
            trace.append((i, pieces(node)))
    return node.levels_completed, trace


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    print("base", pieces(env))
    tests = {
        "large_down": ((13, 13), [2] * 12),
        "large_right": ((13, 13), [4] * 12),
        "horizontal_left": ((35, 52), [3] * 12),
        "horizontal_up": ((35, 52), [1] * 12),
        "vertical_left": ((55, 45), [3] * 12),
        "vertical_up": ((55, 45), [1] * 12),
        "search_best": (
            None,
            [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2
            + [1] * 4 + [3] * 2,
        ),
        "seat_smalls": (
            None,
            [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2
            + [1] * 4 + [3] * 2
            + [(6, 35, 34)] + [3] * 3 + [1] * 3,
        ),
    }
    for name, (click, path) in tests.items():
        print(name, run(env, click, path))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
