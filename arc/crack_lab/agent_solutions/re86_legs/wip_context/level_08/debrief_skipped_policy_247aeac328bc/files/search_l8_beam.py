import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


PAINT = 11


def summary(node):
    grid = arr(node.frame())
    cursor = list(zip(*((grid == 0).nonzero())))
    center = (
        tuple(int(value) for value in cursor[0])
        if len(cursor) == 1
        else (-1, -1)
    )
    count = int((grid == PAINT).sum())
    lower = int((grid[33:, :] == PAINT).sum())
    return center, count, lower


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    horizontal_painted = (
        [5]
        + [2]
        + [3] * 4
        + [2] * 2
        + [1] * 2
        + [1] * 10
        + [3] * 9
        + [1] * 2
    )
    step_many(env, horizontal_painted)

    root = env.clone()
    beam = [(root, [])]
    seen = {arr(root.frame()).tobytes()}
    for depth in range(1, 71):
        candidates = []
        for node, path in beam:
            for action in (1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                frame = arr(child.frame())
                key = frame.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                center, count, lower = summary(child)
                child_path = path + [action]
                if center[1] > 35 and count >= 50:
                    print(
                        "FOUND",
                        child_path,
                        center,
                        count,
                        lower,
                        selected_shape(child),
                        len(seen),
                    )
                    return
                painted = count >= 50
                score = (
                    int(painted) * 10000
                    + center[1] * 30
                    - abs(center[0] - 16)
                    + min(count, 60)
                )
                candidates.append((score, child, child_path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = [(node, path) for _, node, path in candidates[:160]]
        if not beam:
            break
        if depth % 5 == 0:
            center, count, lower = summary(beam[0][0])
            print("PROGRESS", depth, len(seen), center, count, lower)
    print("NOT_FOUND", len(seen))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
