"""Solve level 9 on a reconstructed global board with camera/carrier state."""

from heapq import heappop, heappush
import json
import sys

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


def translated(positions, offset):
    return {(row, col + offset) for row, col in positions}


def successors(state, board, persistent, carrier_bounds):
    pegs, bridges, carrier, camera = state
    occupied = pegs | bridges
    destinations = (board - occupied) | ({carrier} if carrier not in occupied else set())
    pieces = tuple(("peg", position) for position in sorted(pegs))
    pieces += tuple(("bridge", position) for position in sorted(bridges))
    for kind, source in pieces:
        source_col = source[1] - camera
        if not (0 <= source[0] <= 60 and 0 <= source_col <= 60):
            continue
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = source[0] + dr, source[1] + dc
            destination = source[0] + 2 * dr, source[1] + 2 * dc
            destination_col = destination[1] - camera
            if not (0 <= destination[0] <= 60 and 0 <= destination_col <= 60):
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
            child_camera = camera
            if destination == carrier and source != carrier and source[1] != destination[1]:
                child_camera += 20 if destination[1] > source[1] else -20
            macro = (
                (6, source_col + 1, source[0] + 1),
                (6, destination_col + 1, destination[0] + 1),
            )
            yield (
                frozenset(child_pegs), frozenset(child_bridges),
                carrier, child_camera,
            ), macro

    loaded_kind = None
    if carrier in pegs:
        loaded_kind = "peg"
    elif carrier in bridges:
        loaded_kind = "bridge"
    for dc, action in ((-6, 3), (6, 4)):
        next_carrier = carrier[0], carrier[1] + dc
        if not (carrier_bounds[0] <= next_carrier[1] <= carrier_bounds[1]):
            continue
        display_col = next_carrier[1] - camera
        if loaded_kind is None and not (-6 <= display_col <= 66):
            continue
        child_pegs = set(pegs)
        child_bridges = set(bridges)
        child_camera = camera
        if loaded_kind == "peg":
            if next_carrier in occupied:
                continue
            child_pegs.remove(carrier)
            child_pegs.add(next_carrier)
            child_camera += dc
        elif loaded_kind == "bridge":
            if next_carrier in occupied:
                continue
            child_bridges.remove(carrier)
            child_bridges.add(next_carrier)
            child_camera += dc
        yield (
            frozenset(child_pegs), frozenset(child_bridges),
            next_carrier, child_camera,
        ), (action,)


def dense(state):
    pegs = tuple(state[0])
    if len(pegs) <= 1:
        return 0
    pair_terms = sorted(
        abs(first[0] - second[0]) + abs(first[1] - second[1])
        for index, first in enumerate(pegs)
        for second in pegs[index + 1:]
    )
    tree_distance = sum(pair_terms[:len(pegs) - 1])
    alignment = min(
        0 if first[0] == second[0] or first[1] == second[1] else 2
        for index, first in enumerate(pegs)
        for second in pegs[index + 1:]
    )
    return alignment + max(
        0, tree_distance // 6 - (len(pegs) - 1),
    )


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_level = int(env.levels_completed)
    root = env.clone()
    root_holes, root_pegs, root_bridges, _, root_carriers = visible_state(root.frame())
    start_carrier = next(iter(root_carriers))

    phase_two = root.clone()
    for action in PHASE_ONE:
        safe_step(phase_two, action)
    phase_two_state = visible_state(phase_two.frame())
    phase_two_carrier = next(iter(phase_two_state[4]))
    load_pan = start_carrier[1] - phase_two_carrier[1]

    board = set(root_holes | root_pegs | root_bridges)
    revealed_pegs = set()
    revealed_bridges = set()
    persistent = set()
    scan = phase_two.clone()
    max_shift = 0
    for shift in range(40):
        offset = load_pan + 6 * shift
        holes, pegs, bridges, fixed, carriers = visible_state(scan.frame())
        carrier_screen = next(iter(carriers))
        board.update(translated(holes | pegs | bridges, offset))
        revealed_pegs.update(translated(pegs - {carrier_screen}, offset))
        revealed_bridges.update(translated(bridges, offset))
        persistent.update(translated(fixed, offset))
        max_shift = shift
        child = scan.clone()
        safe_step(child, 4)
        if frame_key(child) == frame_key(scan):
            break
        scan = child

    final_bridges = translated(phase_two_state[2], load_pan)
    hidden_bridges = revealed_bridges - final_bridges
    start_pegs = frozenset(set(root_pegs) | revealed_pegs)
    start_bridges = frozenset(set(root_bridges) | hidden_bridges)
    carrier_bounds = (
        start_carrier[1], start_carrier[1] + 6 * max_shift,
    )
    start = (start_pegs, start_bridges, start_carrier, 0)
    print(
        "GLOBAL_WORLD", "pan", load_pan, "max_shift", max_shift,
        "board", len(board), "pegs", tuple(sorted(start[0])),
        "bridges", tuple(sorted(start[1])),
        "persistent", tuple(sorted(persistent)), "carrier_bounds", carrier_bounds,
        flush=True,
    )

    queue = [(dense(start), 0, 0, start)]
    best = {start: 0}
    parent = {}
    serial = 0
    goal = None
    cost_limit = 56
    while queue and len(best) <= 2_000_000:
        _, cost, _, state = heappop(queue)
        if cost != best.get(state):
            continue
        if len(state[0]) == 1:
            goal = state
            break
        if cost >= cost_limit:
            continue
        for child, macro in successors(
            state, frozenset(board), frozenset(persistent), carrier_bounds,
        ):
            child_cost = cost + len(macro)
            if child_cost > cost_limit or child_cost >= best.get(child, 10 ** 9):
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
            if clone.terminal():
                break
            safe_step(clone, action)
        print("GLOBAL_REPLAY", clone.levels_completed, len(actions), visible_state(clone.frame()))
    print("GLOBAL_SEARCH", "states", len(best), "cost", None if goal is None else best[goal])
    print("GLOBAL_ACTIONS", actions)
    print("BASE_LEVEL", base_level)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
