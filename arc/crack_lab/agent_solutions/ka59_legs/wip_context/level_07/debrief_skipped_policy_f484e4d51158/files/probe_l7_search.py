import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


TARGET_H = (25.0, 26.5)
TARGET_V = (47.5, 28.0)
TARGET_LARGE = (40.0, 13.0)


def blobs(env, colors, area=1):
    return connected_components(arr(env.frame())[:63], colors=colors, min_area=area)


def center(blob):
    return blob.centroid


def compact_components(frame, color):
    # np.nonzero returns row/column arrays, so pair them explicitly.
    points = set(zip(*(frame == color).nonzero()))
    out = []
    while points:
        seed = points.pop()
        stack = [seed]
        group = [seed]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in points:
                    points.remove(neighbor)
                    stack.append(neighbor)
                    group.append(neighbor)
        ys = [p[0] for p in group]
        xs = [p[1] for p in group]
        out.append((len(group), max(ys) - min(ys) + 1,
                    max(xs) - min(xs) + 1,
                    sum(ys) / len(group), sum(xs) / len(group)))
    return out


def score(env):
    f = arr(env.frame())[:63]
    value = 0.0
    smalls = [b for b in compact_components(f, 14) if b[0] >= 8]
    horizontal = [b for b in smalls if b[2] > b[1]]
    vertical = [b for b in smalls if b[1] > b[2]]
    if horizontal:
        _, _, _, y, x = max(horizontal)
        value += abs(y - TARGET_H[0]) + abs(x - TARGET_H[1])
    else:
        value += 100
    if vertical:
        _, _, _, y, x = max(vertical)
        value += abs(y - TARGET_V[0]) + abs(x - TARGET_V[1])
    else:
        value += 100
    large = max(compact_components(f, 11), default=None)
    if large:
        _, _, _, y, x = large
        value += abs(y - TARGET_LARGE[0]) + abs(x - TARGET_LARGE[1])
    else:
        value += 100
    return value


def key(env):
    return arr(env.frame())[:63].tobytes()


def choices(env):
    out = [1, 2, 3, 4]
    f = arr(env.frame())[:63]
    selected_y, selected_x = (f == 0).nonzero()
    marker = (
        float(selected_y.mean()), float(selected_x.mean())
    ) if len(selected_y) else (-100.0, -100.0)
    for blob in connected_components(f, colors=(14,), min_area=8):
        y0, x0, y1, x1 = blob.bbox
        y, x = (y0 + y1) / 2, (x0 + x1) / 2
        if abs(y - marker[0]) + abs(x - marker[1]) > 3:
            out.append((6, int(round(x)), int(round(y))))
    return out


def search(env, limit=18000, max_depth=90):
    serial = 0
    root = env.clone()
    q = [(score(root), 0, serial, root, ())]
    seen = {key(root): 0}
    best = (score(root), ())
    expanded = 0
    while q and expanded < limit:
        _, depth, _, node, path = heapq.heappop(q)
        if depth != seen.get(key(node)):
            continue
        expanded += 1
        if node.levels_completed > 6:
            return list(path), expanded, best
        current = score(node)
        if current < best[0]:
            best = (current, path)
            print("best", round(current, 1), "depth", depth, "path", path)
        if depth >= max_depth:
            continue
        for action in choices(node):
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_key = key(child)
            new_depth = depth + 1
            if seen.get(child_key, 10**9) <= new_depth:
                continue
            seen[child_key] = new_depth
            serial += 1
            # Greedy target distance plus a mild path-length pressure.
            priority = score(child) + 0.08 * new_depth
            heapq.heappush(q, (priority, new_depth, serial, child, path + (action,)))
    return None, expanded, best


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    print("stage_score", score(env), flush=True)
    path, expanded, best = search(env, limit=40000, max_depth=120)
    print("search", path, "expanded", expanded, "best", best)
    if path:
        node = env.clone()
        for action in path:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
        print("verified", node.levels_completed, "score", score(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
