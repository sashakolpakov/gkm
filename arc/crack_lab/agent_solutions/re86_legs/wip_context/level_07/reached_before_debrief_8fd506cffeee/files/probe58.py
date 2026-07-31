import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE, bare_frame, descriptor, dot


TARGETS = ((9, 9), (15, 3), (15, 36), (27, 9))
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def desc_key(desc):
    return desc[0], desc[1], desc[2], tuple(sorted(desc[3]))


def segment_distance(point, axes):
    row, col = point
    r0, cr, r1, c0, cc, c1 = axes
    return min(
        abs(col - cc) + max(r0 - row, 0, row - r1),
        abs(row - cr) + max(c0 - col, 0, col - c1),
    )


def search(root, bare, max_states=14000, max_depth=58):
    frame = arr(root.frame())
    start = descriptor(frame, bare, "large-cross", dot(frame))
    start_center = (54, 24)
    queue = [(sum(segment_distance(p, start[2]) for p in TARGETS), 0, 0, root.clone(), (), start_center, start)]
    seen = {desc_key(start): 0}
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
            frame = arr(child.frame())
            desc = descriptor(frame, bare, "large-cross", child_center)
            if desc is None:
                continue
            next_depth = depth + 1
            state = desc_key(desc)
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            distances = tuple(segment_distance(point, desc[2]) for point in TARGETS)
            rank = (-sum(distances), desc[0] == 8)
            if best is None or rank > best[0]:
                best = rank, desc[:3], next_path
                print("BEST", rank, desc[:3], distances, len(next_path), "".join(NAME[a] for a in next_path), flush=True)
            if not any(distances):
                print("GEOMETRY", desc[0], len(next_path), "".join(NAME[a] for a in next_path), desc[:3], len(seen), flush=True)
                if desc[0] == 8:
                    return next_path
            serial += 1
            priority = next_depth + 3 * sum(distances) + (12 if desc[0] != 8 else 0)
            heapq.heappush(queue, (priority, next_depth, serial, child, next_path, child_center, desc))
    print("FAILED", len(seen), best, flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    root.step(USE)
    bare = bare_frame(root)
    solution = search(root, bare)
    if solution:
        print("FULL", (USE, USE) + solution, flush=True)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", run))
