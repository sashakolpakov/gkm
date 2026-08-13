"""Compact clean-room observations for lf52 level 9."""

import importlib.util
import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import (
    action_deltas,
    arr,
    color_counts,
    connected_components,
    frame_delta,
    replay,
    safe_step,
)


def puzzle_delta(before, after):
    """Ignore the harness status row when describing a game-state change."""
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    delta = frame_delta(left, right)
    return delta["count"], delta["bbox"]


def pieces(frame):
    """Locations of compact lattice pieces, omitting static empty holes."""
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 8, 9, 11, 12, 14, 15))
        if blob.color != 9 or blob.size == (4, 4)
    )


def moving_summary(frame):
    blobs = connected_components(frame, colors=(3, 11, 12, 14))
    return tuple((blob.color, blob.bbox, blob.area) for blob in blobs)


def peg_locations(frame):
    return tuple(
        blob.top_left
        for blob in connected_components(frame, colors=(14,))
        if blob.size == (4, 4)
    )


def click_geometry(frame):
    blobs = connected_components(frame, colors=(9, 12, 14))
    return (
        tuple(blob.top_left for blob in blobs
              if blob.color == 14 and blob.size == (4, 4)),
        tuple(blob.top_left for blob in blobs
              if blob.color == 9 and blob.size == (4, 4)),
        tuple(blob.top_left for blob in blobs
              if blob.color == 12 and blob.size == (4, 4)),
    )


def legal_click_moves(root, label):
    frame = root.frame()
    before_geometry = click_geometry(frame)
    candidates = tuple(
        (row, col)
        for row in range(18, 49, 6)
        for col in range(18, 55, 6)
        if row + 1 < 64 and col + 1 < 64
    )
    found = []
    for source in candidates:
        for destination in candidates:
            if source == destination:
                continue
            child = replay(root, (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            ))
            after_geometry = click_geometry(child.frame())
            if (
                after_geometry != before_geometry
                or child.levels_completed > root.levels_completed
            ):
                found.append((
                    source,
                    destination,
                    after_geometry,
                    int(child.levels_completed),
                ))
    print("legal_click_moves", label, found)


def symbolic_solution(frame, max_states=250000):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    cells = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 9, 12, 14)
    )
    pegs = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color == 14
    )
    bridges = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color == 9
    )
    carriers = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color == 12
    )
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (state_pegs, state_bridges), path = queue.popleft()
        if len(state_pegs) == 1 and state_pegs <= carriers:
            return path, len(seen), (cells, carriers)
        occupied = state_pegs | state_bridges
        pieces_to_move = tuple(("peg", cell) for cell in sorted(state_pegs))
        pieces_to_move += tuple(
            ("bridge", cell) for cell in sorted(state_bridges)
        )
        for kind, source in pieces_to_move:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in cells
                    or destination in occupied
                ):
                    continue
                child_pegs = set(state_pegs)
                child_bridges = set(state_bridges)
                if kind == "peg":
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    if midpoint in child_pegs:
                        child_pegs.remove(midpoint)
                else:
                    child_bridges.remove(source)
                    child_bridges.add(destination)
                child = (frozenset(child_pegs), frozenset(child_bridges))
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, path + ((kind, source, destination),)))
    return None, len(seen), (cells, carriers)


def level_nine(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    frame = env.frame()
    print("entry", {
        "levels": int(env.levels_completed),
        "actions": tuple(env.actions),
        "colors": color_counts(frame),
    })
    blobs = connected_components(frame, min_area=4)
    print("blobs", [
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in blobs
    ])
    deltas = action_deltas(env)
    print("key_deltas", {
        action: (delta["count"], delta["bbox"], delta["samples"][:12])
        for action, delta in deltas.items()
    })

    for action in (1, 2, 3, 4, 7):
        clone = env.clone()
        prior = clone.frame()
        observations = []
        for count in range(1, 13):
            safe_step(clone, action)
            current = clone.frame()
            delta = puzzle_delta(prior, current)
            if delta[0] or count == 1:
                observations.append((
                    count,
                    delta,
                    int(clone.levels_completed),
                    pieces(current),
                ))
            prior = current
        print("repeat_key", action, observations)

    coordinate_points = {
        "peg_left": (6, 19, 43),
        "peg_lower_right": (6, 43, 49),
        "carrier": (6, 43, 37),
        "hole": (6, 31, 43),
        "ring": (6, 25, 43),
    }
    for name, action in coordinate_points.items():
        clone = replay(env, (action,))
        print("click", name, puzzle_delta(frame, clone.frame()), pieces(clone.frame()))

    click_sequences = {
        "left_peg_over_ring": ((6, 19, 43), (6, 31, 43)),
        "ring_over_left_peg": ((6, 25, 43), (6, 13, 43)),
        "right_peg_to_carrier": ((6, 43, 49), (6, 43, 37)),
    }
    for name, actions in click_sequences.items():
        clone = replay(env, actions)
        print("click_pair", name, puzzle_delta(frame, clone.frame()),
              int(clone.levels_completed), pieces(clone.frame()))

    queue = deque([(env.clone(), ())])
    initial_key = arr(frame)[1:, :].tobytes()
    seen = {initial_key}
    key_states = []
    while queue and len(seen) < 100:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4, 7):
            child = node.clone()
            before = child.frame()
            safe_step(child, action)
            change = puzzle_delta(before, child.frame())
            if not change[0]:
                continue
            child_key = arr(child.frame())[1:, :].tobytes()
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (action,)
            key_states.append((child_path, change, moving_summary(child.frame())))
            if len(child_path) < 14:
                queue.append((child, child_path))
    print("visible_key_graph", len(seen), key_states)

    legal_click_moves(env, "entry")
    after_bridge = replay(env, ((6, 19, 43), (6, 31, 43)))
    legal_click_moves(after_bridge, "after_left_bridge")

    solution, explored, board = symbolic_solution(frame)
    print("symbolic_solution", explored, solution, board)
    if solution is not None:
        actions = tuple(
            action
            for _, source, destination in solution
            for action in (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
        )
        solved = replay(env, actions)
        print("symbolic_replay", len(actions), int(solved.levels_completed),
              click_geometry(solved.frame()), moving_summary(solved.frame()))
        traced = env.clone()
        for index, (kind, source, destination) in enumerate(solution, 1):
            before = traced.frame()
            safe_step(traced, (6, source[1] + 1, source[0] + 1))
            safe_step(traced, (6, destination[1] + 1, destination[0] + 1))
            print("symbolic_trace", index, kind, source, destination,
                  puzzle_delta(before, traced.frame()),
                  click_geometry(traced.frame()),
                  int(traced.levels_completed))
        for exits in range(1, 6):
            exited = replay(solved, (4,) * exits)
            print("exit_test", exits, int(exited.levels_completed),
                  moving_summary(exited.frame()))


arena.run_program("lf52", level_nine)
