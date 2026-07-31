import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


BOTH_CENTRAL = (
    [3] * 3 + [(6, 35, 52), 4, 3, 4]
    + [3] + [1] * 2 + [4] * 3
)


def rings(env):
    return tuple(
        (b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(14,), min_area=2
        )
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    for hx in (24, 27, 30, 33, 36, 39, 42):
        for vx in (24, 27, 30, 33, 36, 39, 42, 45):
            node = env.clone()
            for action in BOTH_CENTRAL:
                node.step(*action) if isinstance(action, tuple) else node.step(action)
            node.step(6, 46, 29)
            for _ in range(7):
                node.step(3)
            node.step(6, 44, 34)
            for _ in range((42 - hx) // 3):
                node.step(3)
            for _ in range(4):
                node.step(1)
            node.step(6, 25, 29)
            for _ in range((vx - 24) // 3):
                node.step(4)
            before = rings(node)
            for _ in range(8):
                node.step(1)
            after = rings(node)
            if after != before:
                print("h", hx, "v", vx, "before", before, "after", after)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
