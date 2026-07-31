import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


RELAY = [3] * 7 + [(6, 35, 52)] + [1] * 2 + [(6, 34, 29)]


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
    for up in (5, 6, 7):
        for side, horizontal in (("left", [3] * 3), ("right", [4])):
            node = env.clone()
            path = RELAY + [1] * up + horizontal + [(6, 35, 49)]
            for action in path:
                node.step(*action) if isinstance(action, tuple) else node.step(action)
            before = rings(node)
            trace = []
            for index in range(1, 13):
                node.step(4)
                after = rings(node)
                if after != before:
                    trace.append((index, after))
                before = after
            print("y", 27 - 3 * up, side, trace)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
