import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import step_many


TO_UPPER = (
    [5]
    + [2] + [3] * 4 + [2] * 3 + [1] * 2
    + [1] * 11 + [3] * 9
    + [1] * 2
    + [2] * 3 + [4] * 11
    + [1] * 4
)
UPPER = [4] * 2 + [2] + [4] * 2 + [2] + [4] * 2 + [2] * 2 + [3] * 3
LOWER = [2] * 10 + [1] + [3] * 4 + [2] + [3] * 8 + [2]


def cursor(frame):
    points = list(zip(*((frame == 0).nonzero())))
    return tuple(int(value) for value in points[0]) if len(points) == 1 else None


def search(root, target_frame, max_depth, max_states=30000):
    target_key = target_frame.tobytes()
    target_cursor = cursor(target_frame)
    start_frame = arr(root.frame())
    start_cursor = cursor(start_frame)
    start_h = sum(abs(a - b) for a, b in zip(start_cursor, target_cursor)) // 3
    queue = [(start_h, 0, root.clone(), [])]
    seen = {start_frame.tobytes(): 0}
    serial = 0
    expanded = 0
    while queue and len(seen) < max_states:
        _, _, node, path = heappop(queue)
        frame = arr(node.frame())
        expanded += 1
        if frame.tobytes() == target_key:
            return path, len(seen), expanded
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_frame = arr(child.frame())
            key = child_frame.tobytes()
            depth = len(path) + 1
            if seen.get(key, max_depth + 1) <= depth:
                continue
            seen[key] = depth
            child_cursor = cursor(child_frame)
            if child_cursor is None:
                continue
            distance = sum(
                abs(a - b) for a, b in zip(child_cursor, target_cursor)
            ) // 3
            mismatch = int((child_frame != target_frame).sum()) // 64
            serial += 1
            heappush(
                queue,
                (depth + distance + mismatch, serial, child, path + [action]),
            )
    return None, len(seen), expanded


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    root = env.clone()
    step_many(root, TO_UPPER)
    upper_target = root.clone()
    step_many(upper_target, UPPER)
    result = search(root, arr(upper_target.frame()).copy(), len(UPPER) - 1)
    print("upper", len(UPPER), result)

    lower_target = upper_target.clone()
    step_many(lower_target, LOWER)
    result = search(
        upper_target, arr(lower_target.frame()).copy(), len(LOWER) - 1, 50000
    )
    print("lower", len(LOWER), result)


if __name__ == "__main__":
    A.run_program("re86", probe)
