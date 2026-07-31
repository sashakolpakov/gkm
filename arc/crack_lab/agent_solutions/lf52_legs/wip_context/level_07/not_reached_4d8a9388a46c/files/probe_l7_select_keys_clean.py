import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def probe(env):
    setup(env)
    play_lattice_moves(env, (
        ((36, 18), (36, 6)),
        ((30, 6), (42, 6)),
        ((42, 6), (42, 18)),
        ((42, 18), (42, 30)),
        ((36, 6), (36, 18)),
        ((36, 18), (36, 30)),
    ))
    play_actions(env, (2, 3, 3, 3, 2, 2))
    print("CENTER", board(env.frame()))
    for source in ((30, 30), (36, 30), (42, 30)):
        for action in (1, 2, 3, 4):
            child = env.clone()
            child.step(6, source[1] + 1, source[0] + 1)
            child.step(action)
            print("SELECT_KEY", source, action, board(child.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
