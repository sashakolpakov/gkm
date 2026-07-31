import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import frame_delta
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
    play_lattice_moves(env, (((42, 30), (30, 30)),))
    play_actions(env, (1, 1, 4, 4, 4, 2, 3, 3, 3, 3, 3))
    print("CARRIERS_ALIGNED", board(env.frame()))
    before = env.frame()
    play_lattice_moves(env, (((18, 18), (18, 6)),))
    delta = frame_delta(before, env.frame())
    print("TRANSFER", (delta["count"], delta["bbox"]),
          env.levels_completed, board(env.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
