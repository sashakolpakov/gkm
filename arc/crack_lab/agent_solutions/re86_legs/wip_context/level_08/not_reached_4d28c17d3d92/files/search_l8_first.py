import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


def cursor(node):
    points = list(zip(*((arr(node.frame()) == 0).nonzero())))
    if len(points) != 1:
        return None
    return int(points[0][0]), int(points[0][1])


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    env.step(5)
    step_many(env, [1] * 22)
    env.step(5)

    target = (48, 12)
    serial = 0
    start = env.clone()
    queue = [(13, serial, start, [])]
    seen = {arr(start.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 60000:
        _, _, node, path = heappop(queue)
        expanded += 1
        center = cursor(node)
        if center == target:
            shape = selected_shape(node)
            if shape is not None:
                print("AT_TARGET", len(path), shape)
                if shape[2:] == (48, (39, 9, 57, 15)):
                    print("FOUND", path, "seen", len(seen), "expanded", expanded)
                    return
        if len(path) >= 90 or center is None:
            continue
        for action, inverse in ((1, 2), (2, 1), (3, 4), (4, 3)):
            if path and path[-1] == inverse:
                continue
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_center = cursor(child)
            if child_center is None:
                continue
            distance = (
                abs(child_center[0] - target[0])
                + abs(child_center[1] - target[1])
            ) // 3
            serial += 1
            heappush(
                queue,
                (len(path) + 1 + distance, serial, child, path + [action]),
            )
        if expanded % 500 == 0:
            print("PROGRESS", expanded, len(seen), len(queue), len(path))
    print("NOT_FOUND", expanded, len(seen), len(queue))


print("RUN", A.run_program("re86", probe))
