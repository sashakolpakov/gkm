import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE, bare_frame, descriptor
from probe58 import TARGETS, desc_key, segment_distance


DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}
CHAR = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}
CLOSE = tuple(CHAR[ch] for ch in "ULLUUUUUUUULLLUUURRRDDDDUUUURUUUDDDRRDRLULLLUDLLRRRRRDRRLLULLLL")


def point_distance(desc):
    return sum(
        min(abs(row - tr) + abs(col - tc) for row, col in desc[3])
        for tr, tc in TARGETS
    )


def search(root, bare, center, max_states=8000, max_depth=40):
    start = descriptor(arr(root.frame()), bare, "large-cross", center)
    start_distance = point_distance(start)
    queue = [(start_distance, 0, 0, root.clone(), (), center, start)]
    seen = {desc_key(start): 0}
    serial = 0
    best = start_distance, start[:3], ()
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
            desc = descriptor(arr(child.frame()), bare, "large-cross", child_center)
            if desc is None:
                continue
            next_depth = depth + 1
            state = desc_key(desc)
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            value = point_distance(desc)
            if value < best[0]:
                best = value, desc[:3], next_path
                print("BEST", value, desc[:3], len(next_path), "".join(NAME[a] for a in next_path), flush=True)
            if value == 0:
                print("SOLVED", desc[0], len(next_path), "".join(NAME[a] for a in next_path), desc[:3], len(seen), flush=True)
                return next_path
            serial += 1
            heapq.heappush(queue, (next_depth + 4 * value, next_depth, serial, child, next_path, child_center, desc))
    print("FAILED", len(seen), best, flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    root.step(USE)
    bare = bare_frame(root)
    center = (54, 24)
    for action in CLOSE:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    start = descriptor(arr(root.frame()), bare, "large-cross", center)
    point_distances = tuple(
        min(abs(row - tr) + abs(col - tc) for row, col in start[3])
        for tr, tc in TARGETS
    )
    print("START", center, start[:3], point_distances, flush=True)
    suffix = search(root, bare, center)
    if suffix:
        print("FULL", CLOSE + suffix, flush=True)


print("RUN", A.run_program("re86", run))
