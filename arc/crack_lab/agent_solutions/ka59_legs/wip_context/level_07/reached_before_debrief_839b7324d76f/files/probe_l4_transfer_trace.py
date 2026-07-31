import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PATH = (
    [4] * 3 + [2]
    + [(6, 30, 48)] + [2] * 3 + [3] + [1]
    + [(6, 30, 30)] + [1] + [3] * 3 + [2] + [4]
    + [2] * 2 + [4] * 5 + [1] * 7
    + [2] * 3 + [4] * 5
    + [(6, 27, 57)] + [1] + [4] * 4
)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 14), min_area=2
        )
    )


def probe(env):
    for level in range(1, 4):
        getattr(players, f"play_level_{level}")(env)
    previous = pieces(env)
    print("base", previous)
    for index, action in enumerate(PATH, 1):
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        current = pieces(env)
        old_large = tuple(item for item in previous if item[0] == 11)
        new_large = tuple(item for item in current if item[0] == 11)
        if current != previous:
            print(index, action, "large_changed", old_large != new_large, current)
        previous = current


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
