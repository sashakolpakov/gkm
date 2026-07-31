import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_lattice_moves
from perception import connected_components, frame_delta
from probe_l7_relay import setup


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def probe(env):
    setup(env, finish=False)
    print("PRE_WRAP", board(env.frame()))
    print("OBJECTS", tuple(
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in connected_components(
            env.frame(), colors=(1, 8, 11, 12, 14, 15),
        )
        if blob.area < 100
    ))
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
    for source in sorted(bridges | pegs | carriers):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = (source[0] + dr, source[1] + dc)
            if not (0 <= destination[0] < 64 and 0 <= destination[1] < 64):
                continue
            child = env.clone()
            before = child.frame()
            play_lattice_moves(child, ((source, destination),))
            delta = frame_delta(before, child.frame())
            if delta["count"] > 2:
                print("MOVE", source, destination,
                      (delta["count"], delta["bbox"]), board(child.frame()))

    alternate = env.clone()
    play_lattice_moves(alternate, (
        ((18, 46), (30, 46)),
        ((24, 46), (36, 46)),
        ((30, 46), (42, 46)),
        ((42, 46), (42, 58)),
    ))
    print("PEG_WRAP", alternate.levels_completed, board(alternate.frame()))
    slots, carriers, bridges, pegs = _movable_bridge_board(alternate.frame())
    for source in sorted(bridges | pegs):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = (source[0] + dr, source[1] + dc)
            if not (0 <= destination[0] < 64 and 0 <= destination[1] < 64):
                continue
            child = alternate.clone()
            before = child.frame()
            play_lattice_moves(child, ((source, destination),))
            delta = frame_delta(before, child.frame())
            if delta["count"] > 2:
                print("PEG_WRAP_MOVE", source, destination,
                      (delta["count"], delta["bbox"]), board(child.frame()))

    bridge_carrier = env.clone()
    play_lattice_moves(bridge_carrier, (
        ((18, 46), (30, 46)),
        ((24, 46), (36, 46)),
    ))
    before = bridge_carrier.frame()
    play_lattice_moves(bridge_carrier, (((36, 46), (18, 46)),))
    delta = frame_delta(before, bridge_carrier.frame())
    print("LOAD_BRIDGE_IN_OLD_CARRIER",
          (delta["count"], delta["bbox"]), board(bridge_carrier.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
