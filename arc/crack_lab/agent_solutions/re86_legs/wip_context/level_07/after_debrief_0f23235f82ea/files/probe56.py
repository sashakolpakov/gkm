import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE, bare_frame, descriptor


DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}
TARGET_BBOX = (30, 39, 48, 51)


def key(desc):
    return desc[0], desc[1], desc[2], tuple(sorted(desc[3]))


def distance(desc):
    bbox = desc[2]
    return sum(abs(a - b) for a, b in zip(bbox, TARGET_BBOX))


def search(root, bare, max_states=10000, max_depth=52):
    center = (51, 21)
    start = descriptor(arr(root.frame()), bare, "outline", center)
    queue = [(distance(start), 0, 0, root.clone(), (), center, start)]
    seen = {key(start): 0}
    serial = 0
    best = None
    while queue and len(seen) < max_states:
        _, depth, _, node, path, center, old = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            child.step(action)
            dr, dc = DELTA[action]
            child_center = center[0] + dr, center[1] + dc
            if not (0 <= child_center[0] < 63 and 0 <= child_center[1] < 64):
                continue
            desc = descriptor(arr(child.frame()), bare, "outline", child_center)
            if desc is None:
                continue
            next_depth = depth + 1
            state = key(desc)
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            value = distance(desc)
            rank = (desc[0] == 11, -value)
            if best is None or rank > best[0]:
                best = rank, desc[:3], next_path
                print("BEST", rank, desc[:3], len(next_path), "".join(NAME[a] for a in next_path), flush=True)
            if desc[0] == 11 and value == 0:
                print("SOLVED", len(next_path), "".join(NAME[a] for a in next_path), desc[:3], len(seen), flush=True)
                return next_path
            serial += 1
            priority = next_depth + 2 * value + (30 if desc[0] != 11 else 0)
            heapq.heappush(queue, (priority, next_depth, serial, child, next_path, child_center, desc))
    print("FAILED", len(seen), best, flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    bare = bare_frame(root)
    solution = search(root, bare)
    if solution:
        print("FULL", (USE,) + solution, flush=True)


A.run_program("re86", run)
