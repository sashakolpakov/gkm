import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def probe(env):
    setup(env)
    root = board(env.frame())
    print("ROOT", root)
    prefixes = (
        (1,), (1, 3), (1, 3, 3),
        (2,), (2, 3, 3, 3), (2, 3, 3, 2, 2),
        (2, 3), (2, 3, 2, 2, 2),
        (2, 3, 3, 3, 3, 3), (2, 3, 3, 3, 3, 3, 2, 2, 2),
        (2, 4), (2, 4, 2),
    )
    for prefix in prefixes:
        node = env.clone()
        for action in prefix:
            node.step(action)
        print("PREFIX", prefix, board(node.frame()))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            print("BRANCH", prefix, action, board(child.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
