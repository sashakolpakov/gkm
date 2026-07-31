import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_lattice_moves
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def probe(env):
    setup(env)
    play_lattice_moves(env, (
        ((36, 18), (36, 6)),
        ((36, 6), (24, 6)),
        ((30, 6), (18, 6)),
    ))
    print("SHORT_LOADED", board(env.frame()))
    for action in (1, 2, 3, 4):
        child = env.clone()
        for count in range(1, 7):
            child.step(action)
            print("RAY", action, count, board(child.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
