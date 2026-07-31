import importlib.util
import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, _movable_bridge_solution, play_lattice_moves
from legs import play_actions
from perception import arr, color_counts, connected_components, frame_delta


def board(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    return (
        tuple(sorted(slots)),
        tuple(sorted(carriers)),
        tuple(sorted(bridges)),
        tuple(sorted(pegs)),
    )


def compact_blobs(frame):
    return [
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in connected_components(frame, min_area=4)
        if blob.area < 1000
    ]


def frame_key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def find_carrier_path(env, target, max_states=300, max_depth=28):
    queue = deque([(env.clone(), ())])
    seen = {frame_key(env)}
    positions = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        carrier = board(node.frame())[1]
        positions.setdefault(carrier, path)
        if carrier == (target,):
            return path, positions, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = frame_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))
    return None, positions, len(seen)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base = env.frame()
    print("AT", env.levels_completed, "actions", tuple(env.actions))
    print("COLORS", color_counts(base))
    print("BOARD", board(base))
    print("BLOBS", compact_blobs(base))
    for action in env.actions:
        node = env.clone()
        if action == 6:
            node.step(6, 1, 1)
        else:
            node.step(action)
        delta = frame_delta(base, node.frame())
        print("ACTION", action, "delta", (delta["count"], delta["bbox"]),
              "board", board(node.frame()))

    queue = deque([(env.clone(), ())])
    seen = {frame_key(env)}
    carrier_states = {}
    while queue and len(seen) < 120:
        node, path = queue.popleft()
        state = board(node.frame())
        carrier_states.setdefault(state[1], path)
        if len(path) >= 18:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = frame_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))
    print("KEY_GRAPH", len(seen), len(carrier_states))
    for carriers, path in sorted(carrier_states.items()):
        print("CARRIER", carriers, path)

    solution = _movable_bridge_solution(base)
    print("LOCAL_SOLUTION", solution)
    if solution:
        node = env.clone()
        play_lattice_moves(node, solution)
        print("AFTER_LOCAL", node.levels_completed, board(node.frame()))

    node = env.clone()
    play_actions(node, (3, 3, 1, 1, 3, 3, 3))
    print("UNDER_PEG", board(node.frame()))
    before = node.frame()
    play_lattice_moves(node, (((12, 6), (24, 6)),))
    print("LOAD_PEG", frame_delta(before, node.frame()), board(node.frame()))
    print("LOAD_BLOBS", compact_blobs(node.frame()))

    play_actions(node, (4, 4, 4, 2, 2, 3, 3, 2))
    print("PEG_EXIT", board(node.frame()))
    play_lattice_moves(node, (((42, 12), (54, 12)),))
    print("UNLOAD_PEG", board(node.frame()))

    play_actions(node, (1, 4, 4, 1, 1, 4, 4, 4))
    print("UNDER_BRIDGE", board(node.frame()))
    play_lattice_moves(node, (((12, 42), (24, 42)),))
    print("LOAD_BRIDGE", board(node.frame()))
    play_actions(node, (3, 3, 3, 2, 2, 4, 4, 4, 2))
    print("BRIDGE_EXIT", board(node.frame()))
    play_lattice_moves(node, (((42, 42), (54, 42)),))
    print("UNLOAD_BRIDGE", board(node.frame()))

    first_relay = (
        ((54, 12), (54, 24)),
        ((54, 24), (54, 36)),
        ((54, 36), (54, 48)),
        ((54, 42), (54, 54)),
    )
    play_lattice_moves(node, first_relay)
    print("AFTER_WRAP", board(node.frame()))
    play_lattice_moves(node, (((54, 4), (54, 16)),))
    print("AFTER_FIRST_RELAY", node.levels_completed, board(node.frame()))
    print("FIRST_RELAY_BLOBS", compact_blobs(node.frame()))
    second_base = node.frame()
    for action in (1, 2, 3, 4):
        child = node.clone()
        child.step(action)
        delta = frame_delta(second_base, child.frame())
        print("SECOND_ACTION", action, (delta["count"], delta["bbox"]),
              board(child.frame()))
    for path in (
        (3, 3),
        (3, 3, 2),
        (3, 3, 2, 2),
        (3, 3, 2, 2, 2),
        (3, 3, 2, 2, 2, 2),
        (3, 3, 2, 2, 2, 2, 2),
    ):
        child = node.clone()
        play_actions(child, path)
        print("SECOND_PATH", path, board(child.frame()))
    for prefix in (
        (3,), (3, 3), (3, 3, 3), (3, 1),
        (3, 3, 3, 2, 2),
        (3, 3, 3, 2, 2, 4),
        (3, 3, 3, 2, 2, 3),
        (3, 3, 3, 2, 2, 4, 4, 2),
        (3, 1, 1),
        (3, 1, 1, 4, 4, 4),
        (3, 1, 1, 4, 4, 4, 2),
    ):
        for action in (1, 2, 3, 4):
            child = node.clone()
            play_actions(child, prefix + (action,))
            print("SECOND_BRANCH", prefix, action, board(child.frame()))
    child = node.clone()
    for count in range(1, 13):
        child.step(3)
        print("SECOND_LEFT", count, board(child.frame()))

    bridge_carrier_path = (
        3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2,
    )
    child = node.clone()
    play_actions(child, bridge_carrier_path)
    print("SECOND_BRIDGE_CARRIER", board(child.frame()))
    play_lattice_moves(child, (((54, 10), (54, 22)),))
    print("SECOND_LOAD_BRIDGE", board(child.frame()))
    play_lattice_moves(child, (((54, 16), (54, 28)),))
    print("SECOND_PEG_OVER_BRIDGE", board(child.frame()))
    play_actions(child, (1, 3, 3, 1, 1, 4, 4, 4, 2))
    print("SECOND_PAIR_VERTICAL", board(child.frame()))
    before = child.frame()
    play_lattice_moves(child, (((54, 28), (42, 28)),))
    delta = frame_delta(before, child.frame())
    print("SECOND_COLLISION", (delta["count"], delta["bbox"]),
          board(child.frame()))

    child = node.clone()
    play_actions(child, (3, 3, 3, 2, 2))
    for count in range(1, 9):
        child.step(4)
        print("SECOND_MID_RIGHT", count, board(child.frame()))

    child = node.clone()
    play_actions(child, (3, 3, 3, 2, 2, 4, 4))
    for count in range(1, 7):
        child.step(2)
        print("SECOND_MID_DOWN", count, board(child.frame()))

    child = node.clone()
    play_actions(child, (3, 1, 1))
    for count in range(1, 10):
        child.step(4)
        print("SECOND_TOP_RIGHT", count, board(child.frame()))

    child = node.clone()
    play_actions(child, (3, 1, 1, 4, 4, 4))
    for count in range(1, 10):
        child.step(2)
        print("SECOND_RIGHT_DOWN", count, board(child.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
