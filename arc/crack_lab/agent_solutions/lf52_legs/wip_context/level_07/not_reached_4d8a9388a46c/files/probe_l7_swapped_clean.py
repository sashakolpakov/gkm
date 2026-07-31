import json
import sys
from itertools import permutations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import connected_components, frame_delta


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def move(node, source, destination, label):
    before = node.frame()
    play_lattice_moves(node, ((source, destination),))
    delta = frame_delta(before, node.frame())
    print(label, (delta["count"], delta["bbox"]),
          node.levels_completed, board(node.frame()))


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(*action) if isinstance(action, list) else env.step(action)

    # Swap the first chamber's exit ports: peg right, bridge left.
    play_actions(env, (3, 3, 1, 1, 3, 3, 3))
    move(env, (12, 6), (24, 6), "LOAD_PEG_FIRST")
    play_actions(env, (4, 4, 4, 2, 2, 4, 4, 4, 2))
    move(env, (42, 42), (54, 42), "UNLOAD_PEG_RIGHT")

    play_actions(env, (1, 3, 3, 3, 1, 1, 4, 4, 4))
    move(env, (12, 42), (24, 42), "LOAD_BRIDGE_SECOND")
    play_actions(env, (3, 3, 3, 2, 2, 3, 3, 2))
    move(env, (42, 12), (54, 12), "UNLOAD_BRIDGE_LEFT")

    # Wrap with bridge left of peg, then use the same middle carrier.
    for source, destination in (
        ((54, 12), (54, 24)),
        ((54, 24), (54, 36)),
        ((54, 36), (54, 48)),
        ((54, 42), (54, 54)),
        ((54, 4), (54, 16)),
    ):
        move(env, source, destination, "SWAPPED_FIRST_RELAY")

    play_actions(env, (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2))
    move(env, (54, 10), (54, 22), "LOAD_PEG_MIDDLE")
    move(env, (54, 16), (54, 28), "BRIDGE_OVER_PEG_CARGO")
    play_actions(env, (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4))
    move(env, (24, 34), (24, 46), "UNLOAD_PEG_UPPER")

    play_actions(env, (3, 3, 3, 2, 2, 4, 4, 2))
    move(env, (54, 28), (42, 28), "LOAD_BRIDGE_FIXED_PORT")
    play_actions(env, (1, 3, 3, 1, 1, 4, 4, 4, 3, 1, 1, 4, 4, 4, 2))
    print("SWAPPED_PRE_WRAP", board(env.frame()))
    for action in (1, 2, 3, 4):
        child = env.clone()
        for count in range(1, 7):
            child.step(action)
            print("PRE_WRAP_CARGO_RAY", action, count, board(child.frame()))

    # Put peg and bridge on adjacent rows, then reveal the final chamber.
    move(env, (18, 46), (30, 46), "UNLOAD_BRIDGE_OVER_PEG")
    move(env, (24, 46), (36, 46), "PEG_OVER_BRIDGE")
    child = env.clone()
    for count in range(1, 7):
        child.step(2)
        print("PRE_WRAP_DOWN", count, board(child.frame()))
    move(env, (30, 46), (42, 46), "BRIDGE_OVER_PEG")
    for path in ((), (1,), (2,), (3,), (4,)):
        child = env.clone()
        play_actions(child, path)
        play_lattice_moves(child, (((36, 46), (36, 58)),))
        print("PRE_WRAP_KEY", path, board(child.frame()))
    child = env.clone()
    play_lattice_moves(child, (((42, 46), (42, 58)),))
    print("BRIDGE_WRAP_STATE", board(child.frame()))
    move(env, (36, 46), (36, 58), "PEG_WRAP")
    print("SWAPPED_FINAL_ENTRY", board(env.frame()))

    short = env.clone()
    for source, destination in (
        ((36, 18), (36, 6)),
        ((42, 6), (30, 6)),
        ((36, 6), (24, 6)),
        ((30, 6), (18, 6)),
    ):
        move(short, source, destination, "SHORT_BRIDGE_RELAY")
    cooperate = short.clone()
    play_actions(cooperate, (
        4, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2,
    ))
    print("BRIDGE_COOP_CENTER", board(cooperate.frame()))
    portal = short.clone()
    play_actions(portal, (1, 3, 3))
    print("SHORT_BRIDGE_PORTAL", board(portal.frame()))
    print("SHORT_PORTAL_OBJECTS", tuple(
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in connected_components(portal.frame(), colors=(8, 11, 12))
    ))
    for action in (1, 2, 3, 4):
        child = portal.clone()
        for count in range(1, 5):
            child.step(action)
            print("SHORT_PORTAL_RAY", action, count, board(child.frame()))
    edge = short.clone()
    play_actions(edge, (1, 3))
    edge_slots, edge_carriers, _, _ = _movable_bridge_board(edge.frame())
    print("SHORT_BRIDGE_EDGE", board(edge.frame()))
    for source in ((12, 0), (6, 36)):
        for action in (1, 2, 3, 4):
            child = edge.clone()
            child.step(6, source[1] + 1, source[0] + 1)
            child.step(action)
            print("SHORT_EDGE_SELECT_KEY", source, action,
                  board(child.frame()))
    for destination in sorted(edge_slots | edge_carriers):
        child = edge.clone()
        before = board(child.frame())
        play_lattice_moves(child, (((12, 0), destination),))
        after = board(child.frame())
        if after != before:
            print("SHORT_EDGE_MOVE", destination, after)
    for action in (1, 3, 3, 3, 4, 4, 2):
        short.step(action)
        print("SHORT_BRIDGE_ROUTE", action, board(short.frame()))

    # Ship the bridge right, bring the isolated peg back, and capture.
    for action in (
        4, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2,
    ):
        env.step(action)
        print("FINAL_CARRIER_ROUTE", action, board(env.frame()))
    move(env, (36, 18), (36, 30), "PEG_TO_CENTER")
    move(env, (42, 6), (42, 18), "BRIDGE_TO_MIDDLE")
    move(env, (42, 18), (42, 30), "BRIDGE_TO_CENTER")
    child = env.clone()
    move(child, (24, 30), (42, 30), "CARRIER_SELECTS_BRIDGE")
    child = env.clone()
    move(child, (36, 30), (24, 30), "LOAD_PEG_AT_PORT")
    for count in range(1, 5):
        child.step(2)
        print("LOADED_PORT_DOWN", count, board(child.frame()))
    base = board(env.frame())
    for clicks in permutations(((42, 30), (36, 30), (24, 30))):
        child = env.clone()
        for row, column in clicks:
            child.step(6, column + 1, row + 1)
        if board(child.frame()) != base:
            print("THREE_CLICK", clicks, board(child.frame()))
    move(env, (42, 30), (24, 30), "LOAD_BRIDGE_LONG")
    play_actions(env, (1, 4, 4, 4, 2, 4, 2))
    print("BRIDGE_AT_RIGHT_PORT", board(env.frame()))
    move(env, (24, 54), (36, 54), "UNLOAD_BRIDGE_RIGHT")
    move(env, (42, 54), (24, 54), "LOAD_ISOLATED_PEG")
    play_actions(env, (1, 3, 3, 3, 3, 2, 2))
    print("PEG_RETURNED_CENTER", board(env.frame()))
    move(env, (30, 30), (42, 30), "FINAL_CAPTURE")


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
