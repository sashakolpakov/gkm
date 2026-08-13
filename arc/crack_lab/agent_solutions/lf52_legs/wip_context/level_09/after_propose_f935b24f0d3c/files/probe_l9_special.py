"""Probe logical centers of level-9's color-15/color-7 tiles."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


LOAD_MOVES = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def delta(before, after):
    changed = before[1:, :] != after[1:, :]
    ys, xs = changed.nonzero()
    return {
        "changed": int(changed.sum()),
        "bbox": None if not len(ys) else
        (int(ys.min() + 1), int(xs.min()), int(ys.max() + 1), int(xs.max())),
        "transitions": sorted(Counter(
            (int(before[y + 1, x]), int(after[y + 1, x]))
            for y, x in zip(ys, xs)
        ).items()),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    root = env.clone()
    for source, destination in LOAD_MOVES:
        safe_step(root, (6, source[1] + 1, source[0] + 1))
        safe_step(root, (6, destination[1] + 1, destination[0] + 1))

    before = arr(root.frame()).copy()
    probes = {
        "logical_tile": ((6, 59, 25),),
        "tile_center": ((6, 60, 25),),
        "tile_then_7": ((6, 59, 25), 7),
        "tile_double": ((6, 59, 25), (6, 59, 25)),
        "seven_then_tile": (7, (6, 59, 25)),
    }
    for name, path in probes.items():
        node = root.clone()
        for action in path:
            safe_step(node, action)
        print("PROBE", name, path, delta(before, arr(node.frame())),
              {"level": node.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
