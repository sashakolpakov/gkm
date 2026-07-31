import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def shapes(env, colors):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(arr(env.frame())[:63], colors=colors, min_area=2)
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    regions = {
        "c11": (8, 8, 18, 18),
        "c4large": (7, 34, 19, 46),
        "c4horizontal": (22, 22, 31, 28),
        "c4vertical": (25, 43, 31, 52),
    }
    defaults = {}
    for action in (1, 2, 3, 4):
        node = env.clone()
        node.step(action)
        defaults[action] = shapes(node, (4, 11, 14))
    for name, (x0, y0, x1, y1) in regions.items():
        hits = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                for action in (1, 2, 3, 4):
                    node = env.clone()
                    node.step(6, x, y)
                    node.step(action)
                    after = shapes(node, (4, 11, 14))
                    # Ignore the initially selected vertical color-14 piece;
                    # report any command whose non-agent geometry differs.
                    if after != defaults[action]:
                        hits.append((x, y, action, after))
                        break
        print(name, hits[:12], "hit_count", len(hits))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
