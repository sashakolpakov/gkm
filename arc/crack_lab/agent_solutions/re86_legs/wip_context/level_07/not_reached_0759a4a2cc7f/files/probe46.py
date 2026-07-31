import heapq
import json
import sys
import time
from collections import Counter

sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
RECT = [UP] * 3 + [RIGHT] * 4 + [UP] * 8 + [RIGHT] * 9 + [DOWN] * 10
MARK9 = {(6, 12), (9, 9), (9, 30), (27, 12)}
NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}


def geometry(frame):
    points = {
        (int(row), int(col))
        for row, col in zip(*(frame == 9).nonzero())
    } - MARK9
    points |= {
        (int(row), int(col))
        for row, col in zip(*(frame == 0).nonzero())
    }
    rows = Counter(row for row, _ in points)
    cols = Counter(col for _, col in points)
    center_row = rows.most_common(1)[0][0]
    center_col = cols.most_common(1)[0][0]
    vertical = [row for row, col in points if col == center_col]
    horizontal = [col for row, col in points if row == center_row]
    return (
        center_row,
        center_col,
        center_row - min(vertical),
        max(vertical) - center_row,
        center_col - min(horizontal),
        max(horizontal) - center_col,
    )


def heuristic(shape):
    row, col, up, down, left, right = shape
    alignment = (abs(row - 9) + abs(col - 12)) / 3
    arms = (
        abs(up - 3)
        + abs(left - 3)
        + max(0, 18 - down)
        + max(0, 18 - right)
    ) / 3
    return alignment + arms


def search(root, max_states=30000, max_depth=100):
    start_time = time.time()
    frame = arr(root.frame())
    seen = {frame[:63].tobytes()}
    serial = 0
    queue = [(heuristic(geometry(frame)), serial, 0, root.clone(), ())]
    best = (queue[0][0], geometry(frame), ())
    expanded = 0
    while queue and len(seen) < max_states:
        _, _, depth, node, path = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        expanded += 1
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            try:
                child.step(action)
                child_frame = arr(child.frame())
            except Exception:
                continue
            key = child_frame[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            new_path = path + (action,)
            shape = geometry(child_frame)
            if child.levels_completed > 5:
                print('SOLVED', len(new_path), ''.join(NAMES[a] for a in new_path))
                return new_path
            h = heuristic(shape)
            if h < best[0]:
                best = (h, shape, new_path)
                print('best', round(h, 1), shape, len(new_path),
                      ''.join(NAMES[a] for a in new_path))
            serial += 1
            heapq.heappush(
                queue,
                (h + 0.15 * len(new_path), serial, depth + 1, child, new_path),
            )
        if expanded % 2000 == 0:
            print('progress', expanded, len(seen), len(queue),
                  round(time.time() - start_time, 1))
    print('FAILED', expanded, len(seen), 'best', best[:2],
          ''.join(NAMES[a] for a in best[2]))
    return None


def solve(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    for action in RECT:
        root.step(action)
    root.step(USE)
    solution = search(root)
    print('solution', solution)


A.run_program('re86', solve)
