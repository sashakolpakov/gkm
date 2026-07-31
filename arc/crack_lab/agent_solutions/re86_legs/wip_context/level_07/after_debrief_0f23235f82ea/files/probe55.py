import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, SPECS, UP, bare_frame, covered, descriptor, dot


BASE = tuple(
    [UP] * 10
    + [RIGHT, UP, DOWN, DOWN]
    + [RIGHT] * 7
    + [UP]
    + [RIGHT] * 4
    + [UP, DOWN]
)
PAINT = tuple([LEFT] * 7 + [UP] * 2 + [DOWN] * 2 + [RIGHT] * 7)


def desc_key(desc):
    return desc[0], desc[1], desc[2], tuple(sorted(desc[3]))


def segment_distance(point, axes):
    row, col = point
    r0, cr, r1, c0, cc, c1 = axes
    vertical = abs(col - cc) + max(r0 - row, 0, row - r1)
    horizontal = abs(row - cr) + max(c0 - col, 0, col - c1)
    return min(vertical, horizontal)


def search(root, bare, wanted, targets, max_states=24000, max_depth=50):
    frame = arr(root.frame())
    start = descriptor(frame, bare, "small-cross", dot(frame))
    queue = [(0, 0, 0, root.clone(), (), start)]
    seen = {desc_key(start): 0}
    serial = 0
    best = None
    while queue and len(seen) < max_states:
        _, depth, _, node, path, old_desc = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            child.step(action)
            frame = arr(child.frame())
            desc = descriptor(frame, bare, "small-cross", dot(frame))
            if desc is None:
                continue
            next_depth = depth + 1
            key = desc_key(desc)
            if seen.get(key, 999) <= next_depth:
                continue
            seen[key] = next_depth
            next_path = path + (action,)
            distances = tuple(segment_distance(point, desc[2]) for point in targets)
            rank = (desc[0] == wanted, -sum(distances))
            if best is None or rank > best[0]:
                best = rank, desc[:3], next_path
                print("BEST", rank, desc[:3], len(next_path), "".join(NAME[a] for a in next_path), flush=True)
            if desc[0] == wanted and not any(distances):
                print("SOLVED", len(next_path), "".join(NAME[a] for a in next_path), desc[:3], len(seen), flush=True)
                return next_path
            serial += 1
            h = sum(distances) + (18 if desc[0] != wanted else 0)
            heapq.heappush(queue, (next_depth + 3 * h, next_depth, serial, child, next_path, desc))
    print("FAILED", len(seen), best, flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    bare = bare_frame(env)
    root = env.clone()
    for action in BASE + PAINT:
        root.step(action)
    kind, wanted, targets = SPECS[0]
    frame = arr(root.frame())
    desc = descriptor(frame, bare, kind, dot(frame))
    print("START", desc[:3], covered(desc, kind, targets), flush=True)
    suffix = search(root, bare, wanted, targets)
    if suffix:
        print("FULL", BASE + PAINT + suffix, flush=True)


A.run_program("re86", run)
