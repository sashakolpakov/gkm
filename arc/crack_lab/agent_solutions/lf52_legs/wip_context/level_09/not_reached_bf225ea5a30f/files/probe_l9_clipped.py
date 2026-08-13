"""Test valid in-frame pixels of level 9's half-clipped remote peg."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import play_actions, play_lattice_moves
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
    play_actions(root, (4,) * 9)
    before = arr(root.frame())[1:, :].copy()
    for x in (0, 1):
        node = root.clone()
        safe_step(node, (6, x, 13))
        after = arr(node.frame())[1:, :]
        color2 = after == 2
        ys, xs = color2.nonzero()
        print("CLIPPED", {"x": x,
                          "changed": int((before != after).sum()),
                          "destinations": tuple(zip(ys.tolist(),
                                                     xs.tolist()))})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
