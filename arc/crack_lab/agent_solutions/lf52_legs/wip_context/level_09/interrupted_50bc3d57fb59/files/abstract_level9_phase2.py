"""Reconstruct and solve the phase-2 wrapped world in global coordinates."""

from heapq import heappop, heappush
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


PHASE_ONE = (
    (6, 19, 43), (6, 31, 43), (6, 25, 49), (6, 25, 37),
    (6, 25, 43), (6, 25, 31), (6, 25, 37), (6, 25, 25),
    (6, 25, 31), (6, 25, 19), (6, 25, 19), (6, 37, 19),
    (6, 37, 19), (6, 37, 31), (6, 37, 25), (6, 37, 37),
    (6, 37, 31), (6, 37, 43), (6, 37, 37), (6, 37, 49),
    (6, 43, 49), (6, 31, 49), (6, 31, 49), (6, 31, 37),
    (6, 37, 49), (6, 37, 37), (6, 31, 37), (6, 43, 37),
)

COST_LIMIT = 60


def frame_key(env):
    return arr(env.frame())[1:, :].tobytes()


def visible_state(frame):
    blobs = connected_components(frame, colors=(1, 9, 11, 14, 15))
    holes = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    )
    pegs = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    )
    bridges = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    )
    persistent = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    )
    carriers = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in blobs
        if blob.color == 11 and blob.area >= 4
    )
    return holes, pegs, bridges, persistent, carriers


def globalize(positions, shift):
    return {(row, col + 6 * shift) for row, col in positions}


def abstract_successors(state, board, persistent, carrier_screen, max_shift):
    pegs, bridges, shift = state
    carrier = carrier_screen[0], carrier_screen[1] + 6 * shift
    occupied = pegs | bridges
    destinations = (board - occupied) | ({carrier} if carrier not in occupied else set())
    pieces = tuple(("peg", position) for position in sorted(pegs))
    pieces += tuple(("bridge", position) for position in sorted(bridges))
    for kind, source in pieces:
        source_visible_col = source[1] - 6 * shift
        if not (0 <= source[0] <= 60 and 0 <= source_visible_col <= 60):
            continue
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = source[0] + dr, source[1] + dc
            destination = source[0] + 2 * dr, source[1] + 2 * dc
            destination_visible_col = destination[1] - 6 * shift
            if not (0 <= destination[0] <= 60 and 0 <= destination_visible_col <= 60):
                continue
            if midpoint not in occupied | persistent or destination not in destinations:
                continue
            child_pegs = set(pegs)
            child_bridges = set(bridges)
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_bridges.remove(source)
                child_bridges.add(destination)
            macro = (
                (6, source_visible_col + 1, source[0] + 1),
                (6, destination_visible_col + 1, destination[0] + 1),
            )
            yield (frozenset(child_pegs), frozenset(child_bridges), shift), macro

    for offset, action in ((-1, 3), (1, 4)):
        child_shift = shift + offset
        if not (0 <= child_shift <= max_shift):
            continue
        next_carrier = carrier_screen[0], carrier_screen[1] + 6 * child_shift
        child_pegs = set(pegs)
        child_bridges = set(bridges)
        if carrier in child_pegs:
            if next_carrier in occupied:
                continue
            child_pegs.remove(carrier)
            child_pegs.add(next_carrier)
        elif carrier in child_bridges:
            if next_carrier in occupied:
                continue
            child_bridges.remove(carrier)
            child_bridges.add(next_carrier)
        yield (frozenset(child_pegs), frozenset(child_bridges), child_shift), (action,)


def dense(state):
    pegs = tuple(state[0])
    if len(pegs) < 2:
        return 0
    return min(
        (0 if first[0] == second[0] or first[1] == second[1] else 2)
        + max(0, (abs(first[0] - second[0]) + abs(first[1] - second[1])) // 6 - 1)
        for index, first in enumerate(pegs)
        for second in pegs[index + 1:]
    )


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    for action in PHASE_ONE:
        safe_step(env, action)
    base_level = int(env.levels_completed)
    root = env.clone()

    board = set()
    remote_pegs = set()
    bridges = set()
    persistent = set()
    carrier_screen = next(iter(visible_state(root.frame())[4]))
    vertical_changes = []
    scan = root.clone()
    max_shift = 0
    for shift in range(29):
        holes_now, pegs_now, bridges_now, persistent_now, carriers_now = visible_state(scan.frame())
        carrier_now = next(iter(carriers_now))
        board.update(globalize(holes_now, shift))
        board.update(globalize(bridges_now, shift))
        board.update(globalize(pegs_now - {carrier_now}, shift))
        remote_pegs.update(globalize(pegs_now - {carrier_now}, shift))
        bridges.update(globalize(bridges_now, shift))
        persistent.update(globalize(persistent_now, shift))
        available = []
        for action in (1, 2):
            child = scan.clone()
            safe_step(child, action)
            if frame_key(child) != frame_key(scan):
                available.append(action)
        if available:
            vertical_changes.append((shift, tuple(available)))
        max_shift = shift
        child = scan.clone()
        safe_step(child, 4)
        if frame_key(child) == frame_key(scan):
            break
        scan = child

    start_carrier = carrier_screen
    start_pegs = frozenset(remote_pegs | {start_carrier})
    start = (start_pegs, frozenset(bridges), 0)
    queue = [(dense(start), 0, 0, start)]
    best = {start: 0}
    parent = {}
    serial = 0
    goal = None
    while queue and len(best) <= 1_000_000:
        _, cost, _, state = heappop(queue)
        if cost != best.get(state):
            continue
        if len(state[0]) == 1:
            goal = state
            break
        if cost >= COST_LIMIT:
            continue
        for child, macro in abstract_successors(
            state, frozenset(board), frozenset(persistent), carrier_screen, max_shift,
        ):
            child_cost = cost + len(macro)
            if child_cost > COST_LIMIT or child_cost >= best.get(child, 10 ** 9):
                continue
            best[child] = child_cost
            parent[child] = state, macro
            serial += 1
            heappush(queue, (child_cost + dense(child), child_cost, serial, child))

    actions = None
    if goal is not None:
        macros = []
        state = goal
        while state != start:
            state, macro = parent[state]
            macros.append(macro)
        actions = tuple(action for macro in reversed(macros) for action in macro)
        clone = root.clone()
        for action in actions:
            safe_step(clone, action)
        print("PHASE2_ABSTRACT_REPLAY", clone.levels_completed, len(actions), visible_state(clone.frame()))
    print(
        "PHASE2_WORLD", "max_shift", max_shift, "vertical", vertical_changes,
        "board", len(board), "pegs", tuple(sorted(remote_pegs)),
        "bridges", len(bridges), "persistent", len(persistent),
    )
    print("PHASE2_ABSTRACT", "states", len(best), "cost", None if goal is None else best[goal])
    print("PHASE2_ACTIONS", actions)
    print("BASE_LEVEL", base_level)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
