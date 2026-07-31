import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE


TARGETS = ((30, 45), (48, 39), (48, 51))
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def points(node, center):
    frame = arr(node.frame())
    result = {(int(row), int(col)) for row, col in zip(*((frame == 12).nonzero()))}
    if 0 <= center[0] < 63 and 0 <= center[1] < 64 and int(frame[center]) == 0:
        result.add(center)
    return result


def distance(shape):
    return sum(
        min(abs(row - tr) + abs(col - tc) for row, col in shape)
        for tr, tc in TARGETS
    )


def key(shape, center):
    return center, tuple(sorted(shape))


def search(root, max_states=14000, max_depth=58):
    center = (51, 21)
    shape = points(root, center)
    serial = 0
    queue = [(distance(shape), 0, serial, root.clone(), (), center, shape)]
    seen = {key(shape, center): 0}
    best = distance(shape), ()
    while queue and len(seen) < max_states:
        _, depth, _, node, path, center, shape = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            child.step(action)
            dr, dc = DELTA[action]
            child_center = center[0] + dr, center[1] + dc
            if not (0 <= child_center[0] < 63 and 0 <= child_center[1] < 64):
                continue
            child_shape = points(child, child_center)
            if len(child_shape) < 20:
                continue
            next_depth = depth + 1
            state = key(child_shape, child_center)
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            value = distance(child_shape)
            if value < best[0]:
                best = value, next_path
                hits = sum(target in child_shape for target in TARGETS)
                print("BEST", value, hits, child_center, len(child_shape), len(next_path), "".join(NAME[a] for a in next_path), flush=True)
            if all(target in child_shape for target in TARGETS):
                print("SOLVED", len(next_path), "".join(NAME[a] for a in next_path), len(seen), flush=True)
                return next_path
            serial += 1
            heapq.heappush(queue, (next_depth + 3 * value, next_depth, serial, child, next_path, child_center, child_shape))
    print("FAILED", len(seen), best[0], "".join(NAME[a] for a in best[1]), flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    solution = search(root)
    if solution:
        print("FULL", (USE,) + solution, flush=True)


print("RUN", A.run_program("re86", run))
