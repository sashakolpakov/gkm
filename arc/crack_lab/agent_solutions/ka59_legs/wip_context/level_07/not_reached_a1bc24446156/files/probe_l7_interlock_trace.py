import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


STAGE = [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2


def state(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(0, 5, 11, 14), min_area=1
        )
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    for action in STAGE:
        node.step(action)
    print("aligned", state(node))
    for index in range(1, 7):
        node.step(1)
        print("up", index, state(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
