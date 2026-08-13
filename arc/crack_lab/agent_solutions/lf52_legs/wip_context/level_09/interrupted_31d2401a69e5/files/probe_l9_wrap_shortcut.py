"""Test wrapping the remote level-9 bridge pair past world column 118."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import play_actions, play_lattice_moves
from perception import arr, connected_components, safe_step


LOAD = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def pieces(node):
    return tuple(
        (blob.color, blob.top_left, blob.size, blob.area)
        for blob in connected_components(node.frame(), colors=(9, 14))
        if blob.area >= 6
    )


def moved(before, after):
    return int((arr(before)[1:, :] != arr(after)[1:, :]).sum())


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path:
        safe_step(env, action)
    node = env.clone()
    play_lattice_moves(node, LOAD)
    play_actions(node, (4,) * 10)
    print("WRAP_START", pieces(node))

    before = node.frame()
    play_lattice_moves(node, (((18, 46), (18, 58)),))
    print("WRAP_FIRST", {"changed": moved(before, node.frame()),
                         "pieces": pieces(node)})
    safe_step(node, 4)
    print("WRAP_PAN", pieces(node))
    before = node.frame()
    play_lattice_moves(node, (((18, 46), (18, 58)),))
    print("WRAP_SECOND", {"changed": moved(before, node.frame()),
                          "pieces": pieces(node),
                          "level": node.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
