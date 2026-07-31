"""Test exact colored-silhouette coverage as a dense level-2 objective."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def logical_cells(frame, color):
    a = perception.arr(frame)
    return frozenset(
        (r, c)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == color
    )


def probe(env):
    play_level_1(env)
    root = env.clone()
    targets = {color: logical_cells(root.frame(), color) for color in (14, 11, 9)}
    print("sizes", {color: len(cells) for color, cells in targets.items()})
    q = deque([(root.clone(), [])])
    seen = {logical_cells(root.frame(), 0)}
    best = {color: (0, []) for color in targets}
    solved = None
    while q and len(seen) < 18000:
        node, path = q.popleft()
        ours = logical_cells(node.frame(), 0)
        for color, target in targets.items():
            overlap = len(ours & target)
            if overlap > best[color][0]:
                best[color] = (overlap, path)
        if node.levels_completed > 1:
            solved = path
            break
        if len(path) >= 65:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = logical_cells(child.frame(), 0)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    print("states", len(seen), "solved", solved)
    for color in targets:
        score, path = best[color]
        node = perception.replay(root, path)
        print("best", color, score, "/", len(targets[color]), "path", path)
        print("ours", sorted((r // 3, c // 3) for r, c in logical_cells(node.frame(), 0)))
        print("target", sorted((r // 3, c // 3) for r, c in targets[color]))


arena.run_program("cn04", probe)
