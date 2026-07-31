import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PATH = (
    [1] + [3] * 2 + [1] * 2 + [3] * 7 + [1] * 2 + [3] * 3
    + [1] + [3] * 2 + [2] + [3] * 2 + [1] + [4] * 7
    + [2] * 2 + [3] * 2 + [2] + [4] + [1] * 3 + [4] * 7
    + [1] * 3 + [4]
)


def pieces(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(4, 11, 13, 14), min_area=4
        )
    )


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    print("start", pieces(env))
    for action in PATH[:-1]:
        env.step(action)
    print("before_final", env.levels_completed, pieces(env))
    env.step(PATH[-1])
    print("after_final", env.levels_completed)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
