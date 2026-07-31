import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import frame_delta
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def move(node, source, destination, label):
    before = node.frame()
    play_lattice_moves(node, ((source, destination),))
    delta = frame_delta(before, node.frame())
    print(label, (delta["count"], delta["bbox"]),
          node.levels_completed, board(node.frame()))


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
    move(env, (42, 30), (30, 30), "LOAD_PEG_CENTER")

    play_actions(env, (1, 1, 4, 4, 2, 2, 2))
    print("PEG_AT_SIDE", board(env.frame()))
    for action in (1, 2, 3, 4):
        child = env.clone()
        child.step(action)
        print("LOADED_SIDE_KEY", action, board(child.frame()))
    child = env.clone()
    move(child, (36, 30), (36, 54), "BRIDGE_OVER_PEG_CARRIER")
    move(env, (36, 42), (36, 18), "UNLOAD_PEG_OVER_BRIDGE")
    move(env, (36, 18), (36, 6), "PEG_TO_LEFT_SLOT")
    move(env, (36, 30), (36, 18), "BRIDGE_TO_MIDDLE_SLOT")
    move(env, (36, 6), (36, 30), "PEG_OVER_BRIDGE_TO_CENTER")
    move(env, (36, 18), (36, 42), "LOAD_BRIDGE_OVER_PEG")

    play_actions(env, (1, 1, 1, 4, 2, 4, 2))
    print("BRIDGE_AT_RIGHT", board(env.frame()))
    move(env, (24, 54), (36, 54), "UNLOAD_BRIDGE_RIGHT")
    move(env, (42, 54), (24, 54), "LOAD_RIGHT_PEG_OVER_BRIDGE")


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
