"""Test whether the K-room off-grid block persists across stack undo."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_k_room import compact, enter_k_room


def run(root, name, actions):
    child = root.clone()
    print("START", name, compact(child))
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print("STEP", name, index, action, compact(child))


def probe(env):
    enter_k_room(env)
    run(env, "clear_undo2", [(6, 9, 33), 7, 7])
    run(env, "catch_undo2", [(6, 45, 33), 7, 7])


arena.run_program("bp35", probe)
