"""Test border pixels of level 9's loaded carrier as interaction sources."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import play_lattice_moves
from perception import arr, safe_step


LOAD = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path:
        safe_step(env, action)
    root = env.clone()
    play_lattice_moves(root, LOAD)
    before = arr(root.frame())[1:, :].copy()
    for row, col in ((35, 21), (35, 22), (35, 25), (35, 26),
                     (36, 21), (39, 21), (40, 22), (40, 25)):
        node = root.clone()
        safe_step(node, (6, col, row))
        after = arr(node.frame())[1:, :]
        ys, xs = (after == 2).nonzero()
        print("BORDER", {"point": (row, col),
                         "changed": int((before != after).sum()),
                         "dest_count": len(ys),
                         "dest_bbox": None if not len(ys) else
                         (int(ys.min() + 1), int(xs.min()),
                          int(ys.max() + 1), int(xs.max()))})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
