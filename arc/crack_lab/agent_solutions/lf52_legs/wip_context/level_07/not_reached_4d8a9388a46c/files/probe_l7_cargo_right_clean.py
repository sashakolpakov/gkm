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
    attempt(env, (36, 30), (42, 30))
    attempt(env, (42, 30), (36, 30))
    play_actions(env, (2, 3, 3, 3, 2, 2))
    print("CARRIER_CENTER", board(env.frame()))
    child = env.clone()
    for count in range(1, 5):
        child.step(2)
        print("CENTER_DOWN", count, board(child.frame()))
    attempt(env, (30, 30), (36, 30))
    attempt(env, (30, 30), (42, 30))
    play_lattice_moves(env, (((42, 30), (30, 30)),))
    print("PEG_LOADED", board(env.frame()))
    attempt(env, (36, 30), (30, 30))
    attempt(env, (30, 30), (36, 30))
    attempt(env, (36, 30), (24, 30))
    attempt(env, (36, 30), (42, 30))
    for action in (1, 1, 4, 4, 4, 4, 4, 4, 2, 2):
        env.step(action)
        print("PEG_ROUTE", action, board(env.frame()))
    print("PEG_AT_RIGHT", board(env.frame()))
    for source, destination in (
        ((24, 54), (36, 54)),
        ((24, 54), (42, 54)),
        ((24, 54), (48, 54)),
        ((24, 54), (54, 54)),
        ((24, 54), (60, 54)),
        ((42, 54), (24, 54)),
        ((42, 54), (36, 54)),
        ((42, 54), (30, 54)),
    ):
        attempt(env, source, destination)
    for action in (1, 2, 3, 4):
        child = env.clone()
        child.step(action)
        print("CARGO_KEY", action, board(child.frame()))
    play_actions(env, (4, 2))
    print("PEG_AT_GATED_RIGHT", board(env.frame()))
    child = env.clone()
    for count in range(1, 6):
        child.step(2)
        print("LOADED_RIGHT_DOWN", count, board(child.frame()))
    for source, destination in (
        ((24, 54), (36, 54)),
        ((24, 54), (42, 54)),
        ((24, 54), (48, 54)),
        ((24, 54), (54, 54)),
        ((24, 54), (60, 54)),
        ((42, 54), (24, 54)),
        ((42, 54), (36, 54)),
        ((42, 54), (30, 54)),
    ):
        attempt(env, source, destination)
    play_lattice_moves(env, (((24, 54), (36, 54)),))
    print("PAIR_STAGED", env.levels_completed, board(env.frame()))
    for source, destination in (
        ((36, 54), (48, 54)),
        ((36, 54), (54, 54)),
        ((36, 54), (60, 54)),
        ((36, 54), (24, 54)),
        ((42, 54), (30, 54)),
        ((42, 54), (24, 54)),
    ):
        attempt(env, source, destination)
    for action in (1, 2, 3, 4):
        child = env.clone()
        child.step(action)
        print("PAIR_KEY", action, board(child.frame()))
    env.step(2)
    print("CAPTURE_CARRIER", board(env.frame()))
    play_lattice_moves(env, (((42, 54), (30, 54)),))
    print("PORTAL_CAPTURE", env.levels_completed, board(env.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
