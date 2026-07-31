import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def key(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(0, 4, 5, 11, 14), min_area=2
        )
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    frame = arr(env.frame())[:63]
    for row in range(30, 55):
        cols = [col for col in range(0, 25) if int(frame[row, col]) == 2]
        if cols:
            print("wall", row, min(cols), max(cols), len(cols))
    node = env.clone()
    node.step(6, 36, 52)
    for _ in range(7):
        node.step(3)
    print("staged", key(node))
    for count in range(1, 13):
        node.step(1)
        print(count, node.levels_completed, key(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
