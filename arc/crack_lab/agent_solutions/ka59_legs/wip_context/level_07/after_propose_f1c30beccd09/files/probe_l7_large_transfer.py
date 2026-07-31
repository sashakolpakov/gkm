import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PATH = (
    [3] * 7
    + [(6, 35, 52)]
    + [1] * 2
    + [(6, 34, 29)]
    + [1] * 7
    + [3] * 3
    + [2] * 2
)


def movable(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(11, 14), min_area=2
        )
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    for action in PATH:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    print("staged", movable(node))
    previous = movable(node)
    for index in range(1, 13):
        node.step(3)
        current = movable(node)
        print(index, current)
        previous = current


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
