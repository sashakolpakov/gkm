import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def state(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def trace(root, prefix, tick, count=12):
    node = root.clone()
    for item in prefix:
        node.step(*item) if isinstance(item, tuple) else node.step(item)
    print("base", state(node))
    before = state(node)
    for index in range(1, count + 1):
        node.step(tick[index % len(tick)])
        after = state(node)
        old_rings = tuple(x for x in before if x[0] in (11, 14))
        new_rings = tuple(x for x in after if x[0] in (11, 14))
        if new_rings != old_rings:
            print(index, new_rings)
        before = after


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    print("vertical_above_bottom_agent")
    trace(env, [3] * 3 + [(6, 35, 52)], (3, 4))
    print("horizontal_left_bottom_agent")
    trace(env, [(6, 35, 52)] + [4] + [(6, 55, 44)], (1, 2))
    print("vertical_left_top_agent")
    trace(
        env,
        [3] * 7 + [(6, 35, 52)] + [1] * 2
        + [(6, 34, 29)] + [1] * 7 + [4] + [(6, 35, 49)],
        (3, 4),
    )


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
