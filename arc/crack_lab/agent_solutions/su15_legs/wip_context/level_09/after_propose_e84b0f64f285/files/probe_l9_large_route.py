import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _body_groups, _solid_playfield_squares


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


DIRECTIONS = (
    (-8, -8), (-8, 0), (-8, 8), (0, -8),
    (0, 8), (8, -8), (8, 0), (8, 8),
)


def key(env):
    large = _solid_playfield_squares(env, colors=(8,))
    bodies = tuple(
        (color, tuple(_body_groups(env, color))) for color in (7, 14, 13)
    )
    return tuple(blob.bbox for blob in large), bodies


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)

    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    while queue and len(seen) < 2000:
        node, path = queue.popleft()
        large = _solid_playfield_squares(node, colors=(8,))
        if not large or len(path) >= 8:
            continue
        row, col = map(round, large[0].centroid)
        if max(abs(row - 41), abs(col - 11)) <= 1:
            print("ROUTE", path, key(node), flush=True)
            return
        for dr, dc in DIRECTIONS:
            child = node.clone()
            action = (6, col + dc, row + dr)
            child.step(*action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    print("NONE", len(seen), flush=True)


print("RUN", A.run_program("su15", program)[0])
