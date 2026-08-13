"""Search level 9 jointly, before collapsing its local peg board."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def parse(frame, offset):
    blobs = connected_components(frame, colors=(1, 9, 14, 15))
    holes = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    bridges = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    pegs = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1] + offset)
        for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    return holes, bridges, pegs, fixed


def search(cells, fixed, pegs, bridges, max_cost, max_states):
    start = (frozenset(pegs), frozenset(bridges), 22)

    def heuristic(state):
        return 2 * max(0, len(state[0]) - 1)

    serial = 0
    queue = [(heuristic(start), len(pegs), 0, serial, start, ())]
    best = {start: 0}
    expanded = 0
    while queue and len(best) <= max_states:
        _, _, cost, _, state, path = heappop(queue)
        if cost != best.get(state) or cost + heuristic(state) > max_cost:
            continue
        expanded += 1
        state_pegs, state_bridges, carrier_col = state
        if len(state_pegs) == 1:
            return path, len(best), expanded, state

        occupied = state_pegs | state_bridges | fixed
        carrier = (36, carrier_col)
        cargo_kind = (
            "P" if carrier in state_pegs
            else "B" if carrier in state_bridges
            else None
        )
        successors = []

        for action, delta in ((3, -6), (4, 6)):
            child_col = carrier_col + delta
            child_carrier = (36, child_col)
            if not 22 <= child_col <= 106 or child_carrier in occupied:
                continue
            child_pegs = set(state_pegs)
            child_bridges = set(state_bridges)
            if cargo_kind == "P":
                child_pegs.remove(carrier)
                child_pegs.add(child_carrier)
            elif cargo_kind == "B":
                child_bridges.remove(carrier)
                child_bridges.add(child_carrier)
            successors.append((
                1, (action,),
                (frozenset(child_pegs), frozenset(child_bridges), child_col),
            ))

        offset = carrier_col - 22
        for kind, sources in (("P", state_pegs), ("B", state_bridges)):
            for source in sources:
                source_screen = source[1] - offset
                if not 0 <= source_screen <= 60:
                    continue
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr,
                                   source[1] + 2 * dc)
                    destination_screen = destination[1] - offset
                    if (
                        midpoint not in occupied
                        or destination not in cells | {carrier}
                        or destination in occupied
                        or not 0 <= destination_screen <= 60
                    ):
                        continue
                    child_pegs = set(state_pegs)
                    child_bridges = set(state_bridges)
                    if kind == "P":
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    else:
                        child_bridges.remove(source)
                        child_bridges.add(destination)
                    successors.append((
                        2, (kind, source, destination),
                        (frozenset(child_pegs), frozenset(child_bridges),
                         carrier_col),
                    ))

        successors.sort(key=lambda item: (
            len(item[2][0]), item[0], item[1], item[2][2]
        ))
        for step_cost, step, child in successors:
            child_cost = cost + step_cost
            if child_cost + heuristic(child) > max_cost:
                continue
            if child_cost >= best.get(child, 10 ** 9):
                continue
            best[child] = child_cost
            serial += 1
            heappush(queue, (
                child_cost + heuristic(child), len(child[0]), child_cost,
                serial, child, path + (step,),
            ))
        if expanded % 100000 == 0:
            print("joint_progress", expanded, len(best), cost,
                  len(state_pegs), flush=True)
    return None, len(best), expanded, None


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)

    observed = env.clone()
    safe_step(observed, 7)
    initial_holes, initial_bridges, initial_pegs, initial_fixed = parse(
        observed.frame(), 0
    )
    cells = set()
    bridges = set()
    fixed = set()
    scan = observed.clone()
    for index in range(15):
        offset = index * 6
        holes, visible_bridges, _, visible_fixed = parse(scan.frame(), offset)
        cells |= holes | visible_bridges | visible_fixed
        bridges |= visible_bridges
        fixed |= visible_fixed
        safe_step(scan, 4)
    cells |= initial_pegs
    print("joint_world", len(cells), len(initial_pegs), len(bridges),
          len(fixed), tuple(sorted(initial_pegs)), flush=True)

    max_cost = int(os.environ.get("OPT_COST", "56"))
    path, states, expanded, goal = search(
        frozenset(cells), frozenset(fixed), frozenset(initial_pegs),
        frozenset(bridges), max_cost,
        int(os.environ.get("OPT_STATES", "1000000")),
    )
    cost = None if path is None else sum(
        1 if step[0] in (3, 4) else 2 for step in path
    )
    print("joint_model", states, expanded, cost, path, goal, flush=True)
    if path is None:
        return

    replay = env.clone()
    carrier_col = 22
    for step in path:
        if step[0] in (3, 4):
            safe_step(replay, step[0])
            carrier_col += -6 if step[0] == 3 else 6
        else:
            _, source, destination = step
            offset = carrier_col - 22
            move(replay,
                 (source[0], source[1] - offset),
                 (destination[0], destination[1] - offset))
    print("joint_replay", cost, int(replay.levels_completed),
          carrier_col, flush=True)


arena.run_program("lf52", probe)
