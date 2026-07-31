import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import frame_delta
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def attempt(node, source, destination):
    child = node.clone()
    before = child.frame()
    play_lattice_moves(child, ((source, destination),))
    delta = frame_delta(before, child.frame())
    print("TRY", source, destination, (delta["count"], delta["bbox"]),
          child.levels_completed, board(child.frame()))


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

    states = [("CENTER", env.clone())]
    middle = env.clone()
    play_lattice_moves(middle, (((36, 30), (36, 18)),))
    states.append(("MIDDLE", middle))
    left = middle.clone()
    play_lattice_moves(left, (((36, 18), (36, 6)),))
    states.append(("LEFT", left))
    for label, node in states:
        print(label, board(node.frame()))
        for source, destination in (
            ((42, 54), (30, 54)),
            ((42, 54), (24, 54)),
            ((36, 54), (48, 54)),
            ((36, 54), (24, 54)),
        ):
            attempt(node, source, destination)


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
