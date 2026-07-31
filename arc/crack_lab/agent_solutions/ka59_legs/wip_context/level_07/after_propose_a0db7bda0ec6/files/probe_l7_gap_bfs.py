import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from collections import deque
from perception import arr, connected_components


PREFIX = (
    [3] * 3 + [(6, 35, 52), 4, 3, 4]
    + [3] + [1] * 2 + [4] * 3
    + [(6, 46, 29)] + [1] * 2
    + [(6, 44, 34), 1] + [3] * 8
    + [(6, 46, 23)] + [2] * 2 + [3] * 7
)


def key(env):
    frame = arr(env.frame())[:63]
    return (
        tuple(
            (b.color, b.bbox, b.area)
            for b in connected_components(
                frame, colors=(0, 5, 11, 12, 13, 14), min_area=1
            )
        ),
    )


def crossed(env, _):
    selected = connected_components(
        arr(env.frame())[:63], colors=(0,), min_area=1
    )
    return bool(selected) and max(b.bbox[3] for b in selected) < 19


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    for action in PREFIX:
        root.step(*action) if isinstance(action, tuple) else root.step(action)
    queue = deque([(root, [])])
    seen = {key(root)}
    path = None
    while queue and len(seen) < 10000:
        node, prefix = queue.popleft()
        if crossed(node, prefix):
            path = prefix
            break
        if len(prefix) >= 36:
            continue
        frame = arr(node.frame())[:63]
        choices = [1, 2, 3, 4]
        ys, xs = (frame == 5).nonzero()
        if len(ys):
            choices.append((6, int(xs[0]), int(ys[0])))
        for action in choices:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, prefix + [action]))
    print("path", path)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
