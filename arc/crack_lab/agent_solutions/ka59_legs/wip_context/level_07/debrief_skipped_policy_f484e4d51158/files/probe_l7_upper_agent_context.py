import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


STAGE_HORIZONTAL = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
)


def state(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    apply(root, STAGE_HORIZONTAL)
    for up in (4, 5, 6, 7):
        node = root.clone()
        apply(
            node,
            [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * up,
        )
        print("height", 27 - 3 * up, "base", state(node))
        before = state(node)
        for index in range(1, 9):
            node.step(3)
            after = state(node)
            old_rings = tuple(x for x in before if x[0] in (11, 14))
            new_rings = tuple(x for x in after if x[0] in (11, 14))
            if new_rings != old_rings:
                print("left", index, new_rings)
            before = after


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
