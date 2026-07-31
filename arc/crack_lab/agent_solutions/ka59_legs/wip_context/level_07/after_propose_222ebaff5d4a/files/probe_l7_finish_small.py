import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


SOLVE_LARGE = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
    + [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * 6
    + [(6, 34, 47)] + [1] * 2 + [4] * 4 + [1] * 6
    + [3] * 4 + [1] * 4 + [4]
    + [(6, 53, 10)] + [4] * 2 + [3] * 3
    + [(6, 28, 11), 2] + [4] * 3 + [1] + [3] * 2
    + [(6, 53, 10)] + [2] * 6 + [3] * 8 + [2] + [3] * 5
    + [3, 3, 1, 3, 1, 1, 4, 2]
)


def pieces(env):
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
    node = env.clone()
    apply(node, SOLVE_LARGE)
    print("large_solved", pieces(node))

    apply(node, [2] * 3 + [4] * 6 + [1] * 2)
    print("horizontal_target", pieces(node))
    apply(node, [(6, 34, 11)] + [2] * 7)
    print("vertical_above_barrier", pieces(node))
    apply(node, [(6, 26, 25)] + [4] * 3 + [2])
    print("horizontal_above_vertical", pieces(node))
    finish = node.clone()
    previous = pieces(node)
    for index in range(1, 9):
        node.step(2)
        current = pieces(node)
        old_rings = tuple(x for x in previous if x[0] in (11, 14))
        new_rings = tuple(x for x in current if x[0] in (11, 14))
        if new_rings != old_rings:
            print("down", index, new_rings)
        previous = current

    apply(
        finish,
        [2]
        + [(6, 34, 47)] + [3] * 2
        + [(6, 35, 28)] + [3] * 3 + [1],
    )
    print("finished", finish.levels_completed, pieces(finish))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
