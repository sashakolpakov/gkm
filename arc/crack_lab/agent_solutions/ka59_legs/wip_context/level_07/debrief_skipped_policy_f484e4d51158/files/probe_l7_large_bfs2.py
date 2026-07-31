import heapq
import sys
import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PREFIX = [3] * 7 + [(6, 35, 52)] + [1] * 2


def components(env, colors):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=colors, min_area=1
        )
    )


def key(env):
    return arr(env.frame())[:63].tobytes()


def large(env):
    ys, xs = (arr(env.frame())[:63] == 11).nonzero()
    return (
        int(ys.min()),
        int(xs.min()),
        int(ys.max()),
        int(xs.max()),
        int(len(ys)),
    )


def score(env):
    frame = arr(env.frame())[:63]
    ly, lx = (frame == 11).nonzero()
    ry, rx = (frame == 14).nonzero()
    return (
        np.min(np.abs(ry[:, None] - ly[None, :]), initial=100)
        + np.min(np.abs(rx[:, None] - lx[None, :]), initial=100)
    )


def choices(env):
    frame = arr(env.frame())[:63]
    out = [1, 2, 3, 4]
    ys, xs = (frame == 5).nonzero()
    if len(ys):
        out.append((6, int(xs[0]), int(ys[0])))
    return out


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    for action in PREFIX:
        root.step(*action) if isinstance(action, tuple) else root.step(action)
    initial_large = large(root)
    serial = 0
    queue = [(score(root), 0, serial, root, ())]
    seen = {key(root): 0}
    answer = None
    expanded = 0
    while queue and expanded < 30000:
        _, depth, _, node, path = heapq.heappop(queue)
        if depth != seen.get(key(node)):
            continue
        expanded += 1
        if large(node) != initial_large:
            answer = path
            break
        if depth >= 90:
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_key = key(child)
            child_depth = depth + 1
            if seen.get(child_key, 10**9) <= child_depth:
                continue
            seen[child_key] = child_depth
            serial += 1
            priority = score(child) + 0.04 * child_depth
            heapq.heappush(
                queue,
                (priority, child_depth, serial, child, path + (action,)),
            )
    print("expanded", expanded, "path", answer)
    if answer:
        node = root.clone()
        for action in answer:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
        print("large", initial_large, large(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
