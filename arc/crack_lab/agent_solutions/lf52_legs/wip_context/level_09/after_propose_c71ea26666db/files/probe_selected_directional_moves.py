"""Test directional key actions after selecting a legally movable peg."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


def pieces(frame):
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 14))
    ))


def probe(env):
    source = (6, 18, 19)
    for action in (1, 2, 3, 4, 7):
        node = env.clone()
        safe_step(node, source)
        safe_step(node, action)
        print("selected_direction", action, int(node.levels_completed),
              pieces(node.frame()), flush=True)
    for path in ((4, 4, 2, 2), (4, 4, 1, 1), (3, 3, 2, 2)):
        node = env.clone()
        safe_step(node, source)
        for action in path:
            safe_step(node, action)
        print("selected_chain", path, int(node.levels_completed),
              pieces(node.frame()), flush=True)


arena.run_program("lf52", probe)
