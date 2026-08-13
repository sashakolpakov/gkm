"""Search the horizontal level-9 frontier in reproduced world coordinates."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)

ALT_ROW18_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 30), (18, 42)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


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


def search(cells, fixed, pegs, bridges, max_cost=42):
    start = (frozenset(pegs), frozenset(bridges), 22)

    def heuristic(state):
        if os.environ.get("OPT_ASTAR_MODEL") != "1":
            return 0
        state_pegs = tuple(state[0])
        if len(state_pegs) <= 1:
            return 0
        distance = min(
            (abs(first[0] - second[0])
             + abs(first[1] - second[1])) // 6
            for index, first in enumerate(state_pegs)
            for second in state_pegs[index + 1:]
        )
        return max(0, distance - 1)

    serial = 0
    queue = [(heuristic(start), 0, serial, start, ())]
    best = {start: 0}
    while queue:
        _, cost, _, state, path = heappop(queue)
        if cost != best.get(state) or cost > max_cost:
            continue
        state_pegs, state_bridges, carrier_col = state
        if len(state_pegs) == 1 and (
            os.environ.get("OPT_LOADED_GOAL") != "1"
            or (36, carrier_col) in state_pegs
        ):
            return path, len(best), state
        occupied = state_pegs | state_bridges | fixed
        carrier = (36, carrier_col)
        cargo_kind = (
            "P" if carrier in state_pegs
            else "B" if carrier in state_bridges
            else None
        )
        for action, delta in ((3, -6), (4, 6)):
            child_col = carrier_col + delta
            if not 22 <= child_col <= 106 or cost + 1 > max_cost:
                continue
            child_carrier = (36, child_col)
            if child_carrier in occupied:
                continue
            child_pegs = set(state_pegs)
            child_bridges = set(state_bridges)
            if cargo_kind == "P":
                child_pegs.remove(carrier)
                child_pegs.add(child_carrier)
            elif cargo_kind == "B":
                child_bridges.remove(carrier)
                child_bridges.add(child_carrier)
            child = (frozenset(child_pegs), frozenset(child_bridges), child_col)
            child_cost = cost + 1
            if child_cost < best.get(child, 10 ** 9):
                best[child] = child_cost
                serial += 1
                heappush(queue, (child_cost + heuristic(child), child_cost,
                                 serial, child, path + ((action,),)))

        offset = carrier_col - 22
        for kind, sources in (("P", state_pegs), ("B", state_bridges)):
            for source in sources:
                screen_col = source[1] - offset
                if not 0 <= screen_col <= 60:
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
                        or cost + 2 > max_cost
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
                    child = (frozenset(child_pegs),
                             frozenset(child_bridges), carrier_col)
                    child_cost = cost + 2
                    if child_cost >= best.get(child, 10 ** 9):
                        continue
                    best[child] = child_cost
                    serial += 1
                    heappush(queue, (
                        child_cost + heuristic(child), child_cost,
                        serial, child,
                        path + ((kind, source, destination),),
                    ))
    return None, len(best), None


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)
    node = env.clone()
    opening = (ALT_ROW18_RELAY
               if os.environ.get("OPT_OPENING") == "alt_row18"
               else FIRST_RELAY)
    for source, destination in opening:
        move(node, source, destination)

    cells = set()
    bridges = set()
    fixed = set()
    initial_pegs = None
    scan = node.clone()
    for index in range(15):
        offset = index * 6
        holes, visible_bridges, pegs, visible_fixed = parse(
            scan.frame(), offset
        )
        cells |= holes | visible_bridges | pegs | visible_fixed
        bridges |= visible_bridges
        fixed |= visible_fixed
        if index == 0:
            initial_pegs = pegs
        safe_step(scan, 4)
    print("world", len(cells), tuple(sorted(initial_pegs)),
          tuple(sorted(bridges)), tuple(sorted(fixed)), flush=True)
    print("world_edges", tuple(sorted(cell for cell in cells
                                      if cell[0] <= 12 or cell[0] >= 48)),
          flush=True)
    if os.environ.get("OPT_ROWS") == "1":
        print("world_rows", tuple(
            (row, tuple(sorted(col for cell_row, col in cells
                               if cell_row == row)))
            for row in (12, 18, 24, 30, 36, 42)
        ), flush=True)
    path, states, goal = search(
        frozenset(cells), frozenset(fixed),
        frozenset(initial_pegs), frozenset(bridges),
        max_cost=int(os.environ.get("OPT_COST", "42")),
    )
    print("model", states, None if path is None else sum(
        1 if step[0] in (3, 4) else 2 for step in path
    ), path, goal, flush=True)
    if path is not None and os.environ.get("OPT_REPLAY") == "1":
        replay_node = node.clone()
        carrier_col = 22
        for step in path:
            if step[0] in (3, 4):
                safe_step(replay_node, step[0])
                carrier_col += -6 if step[0] == 3 else 6
                continue
            _, source, destination = step
            offset = carrier_col - 22
            move(
                replay_node,
                (source[0], source[1] - offset),
                (destination[0], destination[1] - offset),
            )
        print("model_replay", int(replay_node.levels_completed),
              carrier_col, flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
