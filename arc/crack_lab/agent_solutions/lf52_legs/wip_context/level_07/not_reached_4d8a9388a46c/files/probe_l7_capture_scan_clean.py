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
    play_lattice_moves(env, (((42, 30), (30, 30)),))
    play_actions(env, (1, 1, 4, 4, 4, 2, 4, 2))
    play_lattice_moves(env, (((24, 54), (36, 54)),))

    base_board = board(env.frame())
    print("PAIR", env.levels_completed, base_board)
    hits = []
    for y in range(22, 36):
        for x in range(53, 59):
            child = env.clone()
            child.step(6, 55, 43)
            child.step(6, x, y)
            child_board = board(child.frame())
            if child.levels_completed != env.levels_completed or child_board != base_board:
                hits.append((x, y, child.levels_completed, child_board))
    print("HITS", tuple(hits))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
