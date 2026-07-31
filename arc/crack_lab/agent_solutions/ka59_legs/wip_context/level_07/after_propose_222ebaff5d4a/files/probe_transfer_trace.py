import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


LEVEL_6_PATH = (
    [1] + [3] * 2 + [1] * 2 + [3] * 7 + [1] * 2 + [3] * 3
    + [1] + [3] * 2 + [2] + [3] * 2 + [1] + [4] * 7
    + [2] * 2 + [3] * 2 + [2] + [4] + [1] * 3 + [4] * 7
    + [1] * 3 + [4]
)


def pieces(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(11, 13, 14), min_area=4
        )
    )


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    previous = pieces(env)
    print("base", previous)
    for index, action in enumerate(LEVEL_6_PATH, 1):
        env.step(action)
        current = pieces(env)
        old_large = tuple(p for p in previous if p[0] == 11)
        new_large = tuple(p for p in current if p[0] == 11)
        old_small = tuple(p for p in previous if p[0] == 14)
        new_small = tuple(p for p in current if p[0] == 14)
        if new_large != old_large or new_small != old_small:
            print(index, action, "small", new_small, "large", new_large)
        previous = current


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
