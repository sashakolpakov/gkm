import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


RELAY_VERTICAL = [3] * 7 + [(6, 35, 52)] + [1] * 2 + [(6, 34, 29)]


def state(env):
    blobs = connected_components(
        arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
    )
    return tuple((b.color, b.bbox, b.area) for b in blobs)


def replay(root, prefix, action, count=8):
    node = root.clone()
    for item in prefix:
        node.step(*item) if isinstance(item, tuple) else node.step(item)
    print("base", state(node))
    previous = state(node)
    for index in range(1, count + 1):
        node.step(action)
        current = state(node)
        ring_before = tuple(x for x in previous if x[0] in (11, 14))
        ring_after = tuple(x for x in current if x[0] in (11, 14))
        if ring_after != ring_before:
            print(index, ring_after)
        previous = current


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    for up in (5, 6, 7):
        print("right_contact_y", 27 - 3 * up)
        replay(env, RELAY_VERTICAL + [1] * up + [4], 4)
    for up in (5, 6, 7):
        print("left_contact_y", 27 - 3 * up)
        replay(env, RELAY_VERTICAL + [1] * up + [3] * 3, 3)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
