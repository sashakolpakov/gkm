import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import frame_delta
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def attempt(node, source, destination, label):
    child = node.clone()
    before = child.frame()
    play_lattice_moves(child, ((source, destination),))
    delta = frame_delta(before, child.frame())
    print(label, (delta["count"], delta["bbox"]), board(child.frame()))
    return child


def probe(env):
    setup(env)
    print("ROOT", board(env.frame()))
    attempt(env, (36, 24), (36, 36), "MOVE_FIXED_SUPPORT")
    wrapped = env.clone()
    play_lattice_moves(wrapped, (((36, 18), (36, 6)),))
    attempt(wrapped, (36, 6), (36, 54), "BRIDGE_WRAP_LEFT_TO_RIGHT")

    left = env.clone()
    play_actions(left, (2, 3, 3, 3, 3, 3, 2))
    print("LEFT_PORT", board(left.frame()))
    loaded = attempt(left, (36, 18), (24, 18), "LOAD_BRIDGE_LEFT")

    play_actions(loaded, (1, 4, 4, 4, 4, 4, 4, 2))
    print("BRIDGE_AT_RIGHT_PORT", board(loaded.frame()))
    delivered = attempt(loaded, (24, 54), (36, 54),
                        "UNLOAD_BRIDGE_RIGHT")

    attempt(delivered, (42, 54), (30, 54), "PEG_OVER_BRIDGE_UP")
    attempt(delivered, (42, 54), (24, 54), "PEG_TO_CARRIER_DIRECT")
    attempt(delivered, (36, 54), (48, 54), "BRIDGE_OVER_PEG_DOWN")
    attempt(delivered, (36, 54), (24, 54), "RELOAD_BRIDGE_RIGHT")

    center = env.clone()
    play_actions(center, (2, 3, 3, 3, 2, 2))
    print("CENTER_PORT", board(center.frame()))
    bridge_center = attempt(center, (36, 18), (36, 30),
                            "BRIDGE_TO_CENTER_SLOT")
    attempt(bridge_center, (36, 30), (30, 30),
            "BRIDGE_TO_CENTER_CARRIER")

    side = env.clone()
    play_actions(side, (2, 3, 2, 2, 2))
    print("SIDE_PORT", board(side.frame()))
    bridge_side = attempt(side, (36, 18), (36, 30),
                          "BRIDGE_TO_SIDE_SLOT")
    attempt(bridge_side, (36, 30), (36, 42),
            "BRIDGE_TO_SIDE_CARRIER")
    attempt(bridge_side, (36, 30), (36, 54),
            "BRIDGE_OVER_SIDE_CARRIER")
    for action in (1, 2, 3, 4):
        child = bridge_side.clone()
        child.step(action)
        print("SIDE_WITH_BRIDGE_KEY", action, board(child.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
