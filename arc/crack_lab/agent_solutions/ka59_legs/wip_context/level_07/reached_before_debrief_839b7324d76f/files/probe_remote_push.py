import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PREFIX = (
    [1] + [3] * 2 + [1] * 2 + [3] * 7 + [1] * 2 + [3] * 3
    + [1] + [3] * 2 + [2] + [3] * 2 + [1]
)


def state(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(11, 14), min_area=2
        )
    )


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    for action in PREFIX:
        env.step(action)
    print("base", state(env))
    for index in range(1, 7):
        env.step(1)
        print(index, state(env))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
