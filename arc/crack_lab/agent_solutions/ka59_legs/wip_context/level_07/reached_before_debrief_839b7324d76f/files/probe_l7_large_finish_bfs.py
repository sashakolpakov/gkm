import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PREFIX = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
    + [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * 6
    + [(6, 34, 47)] + [1] * 2 + [4] * 4 + [1] * 6
    + [3] * 4 + [1] * 4 + [4]
    + [(6, 53, 10)] + [4] * 2 + [3] * 3
    + [(6, 28, 11), 2] + [4] * 3 + [1] + [3] * 2
    + [(6, 53, 10)] + [2] * 6 + [3] * 8 + [2] + [3] * 5
)


def large_top(env):
    blobs = connected_components(
        arr(env.frame())[:63], colors=(11,), min_area=4
    )
    return blobs[0].bbox[0] if blobs else -1


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def key(env):
    return arr(env.frame())[:63].tobytes()


def choices(env):
    out = [1, 2, 3, 4]
    frame = arr(env.frame())[:63]
    ys, xs = (frame == 5).nonzero()
    if len(ys):
        out.append((6, int(xs[0]), int(ys[0])))
    return out


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    apply(root, PREFIX)
    print("root_top", large_top(root))
    queue = deque([(root, ())])
    seen = {key(root)}
    answer = None
    while queue and len(seen) < 12000:
        node, path = queue.popleft()
        if len(path) >= 72:
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + (action,)
            if large_top(child) > 21:
                answer = child_path
                queue.clear()
                break
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    print("expanded", len(seen), "path", answer)
    if answer:
        child = root.clone()
        for index, action in enumerate(answer, 1):
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            print("trace", index, action, pieces(child))
        print("verified", large_top(child), child.levels_completed)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
