import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


STAGE_HORIZONTAL = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def rings(env):
    return tuple(item for item in pieces(env) if item[0] in (11, 14))


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    apply(node, STAGE_HORIZONTAL)
    print("h_central", rings(node))
    apply(node, [(6, 35, 34), 1] + [3] * 7 + [1] * 3)
    print("h_contact", rings(node))

    # Put the vertical ring above the lower cycling agent and let the agent
    # hand it into the central region while it moves right.
    apply(node, [(6, 34, 47)] + [1] * 2 + [4] * 4)
    print("v_above_agent", rings(node))

    # Check every three-pixel alignment of each complementary contact.  The
    # attempts toward the large piece are blocked, so repeating them is also
    # a compact way to cover every six-phase agent timing.
    initial_large = tuple(item for item in rings(node) if item[0] == 11)
    hits = []
    for vertical_up in (5, 6, 7):
        vertical_y = 30 - 3 * vertical_up
        vertical = node.clone()
        apply(
            vertical,
            [(6, 46, 32)] + [3] * 4 + [1] * vertical_up + [3] * 3,
        )
        for horizontal_shift in (-1, 0, 1):
            horizontal_x = 12 + 3 * horizontal_shift
            aligned = vertical.clone()
            shift = [3] * -horizontal_shift if horizontal_shift < 0 else \
                    [4] * horizontal_shift
            apply(aligned, [(6, 14, 22)] + shift)
            print("contacts", vertical_y, horizontal_x, rings(aligned))
            for first in ("vertical", "horizontal"):
                for a in range(1, 7):
                    for b in range(1, 7):
                        if first == "vertical":
                            path = (
                                [(6, 25, vertical_y + 2)] + [3] * a
                                + [(6, horizontal_x + 2, 22)] + [1] * b
                                + [2, 2]
                                + [(6, 25, vertical_y + 2), 4, 4]
                            )
                        else:
                            path = (
                                [(6, horizontal_x + 2, 22)] + [1] * a
                                + [(6, 25, vertical_y + 2)] + [3] * b
                                + [4, 4]
                                + [(6, horizontal_x + 2, 22), 2, 2]
                            )
                        child = aligned.clone()
                        apply(child, path)
                        large = tuple(
                            item for item in rings(child) if item[0] == 11
                        )
                        if large != initial_large:
                            hits.append(
                                (vertical_y, horizontal_x, first, a, b,
                                 path, large)
                            )
    print("hits", hits)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
