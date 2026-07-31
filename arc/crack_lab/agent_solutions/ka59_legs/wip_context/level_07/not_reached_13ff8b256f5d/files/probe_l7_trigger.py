import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def large(env):
    blobs = connected_components(
        arr(env.frame())[:63], colors=(11,), min_area=4
    )
    return tuple((b.bbox, b.area) for b in blobs)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    base = large(env)
    tests = (
        ("v_left", None, [3] * 12),
        ("v_down_left", None, [2] * 3 + [3] * 12),
        ("v_up_left", None, [1] * 3 + [3] * 12),
        ("h_right", (36, 52), [4] * 12),
        ("h_up_right", (36, 52), [1] * 3 + [4] * 12),
        ("h_down_right", (36, 52), [2] * 3 + [4] * 12),
    )
    for name, click, path in tests:
        for delay in range(6):
            node = env.clone()
            for _ in range(delay):
                node.step(4)
            if click:
                node.step(6, *click)
            changed = None
            for index, action in enumerate(path, 1):
                node.step(action)
                if large(node) != base:
                    changed = (index, large(node))
                    break
            print(name, delay, changed)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
