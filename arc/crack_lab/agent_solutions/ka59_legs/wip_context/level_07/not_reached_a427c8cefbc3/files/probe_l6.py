import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, bounded_bfs, color_counts, connected_components


def delta(before, after):
    a, b = arr(before)[:63], arr(after)[:63]
    ys, xs = (a != b).nonzero()
    if not len(ys):
        return (0, None, ())
    transitions = {}
    for y, x in zip(ys, xs):
        pair = (int(a[y, x]), int(b[y, x]))
        transitions[pair] = transitions.get(pair, 0) + 1
    return (
        int(len(ys)),
        (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        tuple(sorted(transitions.items())),
    )


def components(frame):
    counts = color_counts(arr(frame)[:63])
    background = max(counts, key=counts.get)
    blobs = connected_components(arr(frame)[:63], min_area=2)
    return background, [
        (b.color, b.bbox, b.area)
        for b in blobs
        if b.color != background
    ]


def pieces(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=4)
    ]


def run_path(env, path):
    node = env.clone()
    trace = []
    for i, action in enumerate(path, 1):
        node.step(action)
        if i == 1 or i == len(path) or i % 3 == 0:
            trace.append((i, action, pieces(node), node.levels_completed))
    return node, trace


def macro_bfs(env, max_states=1500, max_depth=18):
    def key(node):
        return tuple(
            (b.color, b.bbox, b.area)
            for b in connected_components(
                arr(node.frame())[:63], colors=(11, 13, 14), min_area=4
            )
        )

    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    best = (10**9, (), key(env))
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        current = key(node)
        boxes = {color: bbox for color, bbox, _ in current if color in (11, 14)}
        if 11 in boxes and 14 in boxes:
            y0, x0, y1, x1 = boxes[11]
            large_distance = abs((x0 + x1) // 2 - 7) + abs((y0 + y1) // 2 - 7)
            y0, x0, y1, x1 = boxes[14]
            small_distance = abs((x0 + x1) // 2 - 49) + abs((y0 + y1) // 2 - 22)
            score = large_distance + small_distance
            if score < best[0]:
                best = (score, path, current)
        if node.levels_completed > 5:
            return [action for macro in path for action in (macro,) * 3], len(seen), best
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child_path = path + (action,)
            child = node.clone()
            for _ in range(3):
                child.step(action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            if child.levels_completed > 5:
                return [a for macro in child_path for a in (macro,) * 3], len(seen), best
            queue.append((child, child_path))
    return None, len(seen), best


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    base = arr(env.frame()).copy()
    background, blobs = components(base)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(base[:63]), "background", background)
    print("components", blobs)
    print("arrows", {
        action: delta(base, (lambda c: (c.step(action), c.frame())[1])(env.clone()))
        for action in (1, 2, 3, 4)
    })
    candidates = []
    for color, bbox, area in blobs:
        y0, x0, y1, x1 = bbox
        if area >= 4 and (x1 - x0) <= 20 and (y1 - y0) <= 20:
            candidates.append((x0, y0, color, bbox))
    for x, y, color, bbox in candidates:
        selected = env.clone()
        before = arr(selected.frame()).copy()
        selected.step(6, x, y)
        effects = {}
        for action in (1, 2, 3, 4):
            moved = selected.clone()
            pre_move = arr(moved.frame()).copy()
            moved.step(action)
            effects[action] = delta(pre_move, moved.frame())
        print("select", (x, y, color, bbox), delta(before, selected.frame()), effects)
    routes = {
        "gap": [3] * 7 + [1] * 9 + [4] * 6,
        "direct": [1] * 9 + [3] + [4],
        "left15": [3] * 15,
        "left_up20": [3] * 7 + [1] * 20,
        "left_wait_left": [3] * 7 + [2] * 6 + [3] * 6,
        "center_gap": [3] * 9 + [1] * 9 + [4] * 6,
        "center_up15": [3] * 9 + [1] * 15,
        "up30": [1] * 30,
        "up12_left": [1] * 12 + [3],
        "left15_up12": [3] * 15 + [1] * 12,
        "left30": [3] * 30,
    }
    for name, path in routes.items():
        node, trace = run_path(env, path)
        print("route", name, "won", node.levels_completed > 5, "trace", trace)
        if name == "center_up15":
            f = arr(node.frame())
            print("gap_crop", [
                (y, "".join("0123456789ABCDEF"[int(v)] for v in f[y, 24:40]))
                for y in range(27, 41)
            ])
    path = bounded_bfs(
        env,
        lambda node, _: node.levels_completed > 5,
        actions=(1, 2, 3, 4),
        key_fn=lambda node: tuple(pieces(node)),
        max_states=1,
        max_depth=50,
    )
    print("bfs", path)
    if path:
        node, trace = run_path(env, path)
        print("bfs_verified", node.levels_completed > 5, "trace", trace)
    macro_path, states, best = macro_bfs(env)
    print("macro_bfs", macro_path, "states", states, "best", best)
    if macro_path:
        node, trace = run_path(env, macro_path)
        print("macro_verified", node.levels_completed > 5, "trace", trace)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
