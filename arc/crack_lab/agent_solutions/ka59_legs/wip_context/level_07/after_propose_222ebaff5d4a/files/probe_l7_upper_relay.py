import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


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

    # Hand the vertical ring into the central room, then place it against
    # the right wall of the sealed upper-middle lane.
    apply(
        node,
        [3] * 7
        + [(6, 35, 52)]
        + [1] * 2
        + [(6, 34, 29)]
        + [1] * 5
        + [4],
    )
    print("vertical_middle", pieces(node))

    # Take the horizontal ring out through the lower-right opening and up
    # the exterior lane.  Vary the phase on a blocked boundary so arrival
    # below the upper cycling agent is tested at all six timings.
    for delay in range(6):
        child = node.clone()
        apply(
            child,
            [(6, 35, 49)]
            + [1] * 2
            + [4] * 8
            + [4] * delay
            + [3] * 2
            + [1] * 10
            + [3] * 2,
        )
        print("upper_right", delay, pieces(child))
        previous = pieces(child)
        for index in range(1, 9):
            child.step(3)
            current = pieces(child)
            old_movable = tuple(x for x in previous if x[0] in (11, 14))
            new_movable = tuple(x for x in current if x[0] in (11, 14))
            if new_movable != old_movable:
                print("left", delay, index, new_movable)
            previous = current


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
