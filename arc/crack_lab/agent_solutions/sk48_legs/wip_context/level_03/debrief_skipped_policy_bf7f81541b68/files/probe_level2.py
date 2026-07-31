import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
import numpy as np
from perception import (
    action_deltas,
    bounded_bfs,
    color_counts,
    frame_delta,
    level_goal,
    object_candidates,
    replay,
)


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame)
        if o["color"] != 1
    ]


def probe(env):
    players.play_level_1(env)
    print("LEVEL", env.levels_completed, "ACTIONS", tuple(env.actions))
    print("COUNTS", color_counts(env.frame()))
    print("OBJECTS", compact_objects(env.frame()))
    frame = np.asarray(env.frame())
    chars = "0123456789ABCDEF"
    print("MAP")
    for r in range(0, 64, 4):
        line = ""
        for c in range(0, 64, 4):
            vals, counts = np.unique(frame[r:r+4, c:c+4], return_counts=True)
            line += chars[int(vals[int(np.argmax(counts))])]
        print(line)
    for action, delta in action_deltas(env, tuple(env.actions)).items():
        print("DELTA", action, delta["count"], delta["bbox"])
    base = env.frame()
    paths = [
        *([1, 1, 1] + [4] * 3 + [3] * 3 + [1] * n
          for n in range(1, 7)),
        *([1, 1, 1] + [4] * 3 + [3] * 3 + [1] * n + [2]
          for n in range(3, 7)),
    ]
    for path in paths:
        node = replay(env, path)
        objects = [
            (o["color"], o["bbox"][:2])
            for o in object_candidates(node.frame())
            if o["color"] in (8, 9, 12, 14) and o["bbox"][0] < 53
        ]
        delta = frame_delta(base, node.frame())
        print("PATH", "".join(map(str, path)), "L", node.levels_completed,
              "D", delta["count"], delta["bbox"], "O", objects)
    for x, y in ((7, 44), (31, 26), (37, 26), (43, 26), (49, 26),
                 (19, 58), (25, 58), (31, 58), (37, 58), (43, 58)):
        node = env.clone()
        try:
            node.step(6, x, y)
            delta = frame_delta(base, node.frame())
            print("CLICK", (x, y), delta["count"], delta["bbox"])
        except Exception as exc:
            print("CLICK_ERR", type(exc).__name__)
            break


levels, path, err = arena.run_program("sk48", probe)
print("DONE", levels, len(path), err)
