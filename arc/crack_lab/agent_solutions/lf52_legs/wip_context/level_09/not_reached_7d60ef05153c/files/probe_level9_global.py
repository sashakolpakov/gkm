"""Joint symbolic search over every phase-one carrier-entry arrangement."""

import json
from collections import deque
from heapq import heappop, heappush

import gkm_try
from perception import connected_components


WORLD_HOLES = {
    (12, 58), (12, 64), (12, 76), (12, 88), (12, 94), (12, 100), (12, 106), (12, 112), (12, 118),
    (18, 4), (18, 16), (18, 22), (18, 52), (18, 58), (18, 64), (18, 70), (18, 76), (18, 82),
    (18, 88), (18, 94), (18, 100), (18, 118), (24, 10), (24, 16), (24, 22), (24, 52),
    (24, 64), (24, 70), (24, 76), (24, 82), (24, 88), (24, 100), (24, 112), (24, 118),
    (30, 4), (30, 10), (30, 16), (30, 22), (30, 112), (30, 118),
    (36, 4), (36, 10), (36, 112), (36, 118), (42, 4), (42, 10),
    (48, 4), (48, 10), (48, 16), (48, 22),
}
WORLD_FIXED = {(12, 70), (12, 82), (24, 58), (24, 94), (24, 106)}
WORLD_TARGET = {(12, 52)}
WORLD_FAR_BRIDGES = {(18, 106), (18, 112)}
CARRIER_POSITIONS = tuple((36, column) for column in range(22, 107, 6))
PHASE_CARRIERS = ((36, 42), (36, 48), (36, 54), (36, 60))


def pieces(frame):
    blobs = connected_components(frame, colors=(1, 9, 11, 12, 14))
    border_positions = {
        (blob.bbox[0] + 1, blob.bbox[1] + 1)
        for blob in blobs if blob.color == 11 and blob.area >= 4
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    } | border_positions
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4)
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    return holes, pegs, bridges, carriers


def phase_one_goals(frame):
    holes, root_pegs, root_bridges, carriers = pieces(frame)
    carrier = next(iter(carriers))
    slots = frozenset(holes | root_pegs | root_bridges | carriers)
    start = (frozenset(root_pegs), frozenset(root_bridges))
    queue = deque([start]); paths = {start: ()}; goals = []
    while queue:
        pegs, bridges = queue.popleft(); path = paths[(pegs, bridges)]
        if carrier in pegs:
            goals.append((pegs, bridges, path))
            continue
        occupied = pegs | bridges
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                if midpoint not in occupied or destination not in slots or destination in occupied:
                    continue
                child_pegs = set(pegs); child_bridges = set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = frozenset(child_pegs), frozenset(child_bridges)
                if child in paths: continue
                paths[child] = path + ((kind, source, destination),)
                queue.append(child)
    return slots, carrier, goals, len(paths)


def moving_phase_goals(frame):
    holes, root_pegs, root_bridges, carriers = pieces(frame)
    start_carrier = next(iter(carriers))
    base_slots = frozenset(holes | root_pegs | root_bridges)
    start = start_carrier, frozenset(root_pegs), frozenset(root_bridges)
    distance = {start: 0}; paths = {start: ()}; queue = [(0, 0, start)]; serial = 1; goals = []
    while queue:
        cost, _, state = heappop(queue)
        if cost != distance[state]: continue
        carrier, pegs, bridges = state
        if carrier in pegs:
            goals.append((cost, carrier, pegs, bridges, paths[state])); continue
        carrier_index = PHASE_CARRIERS.index(carrier)
        for action, next_index in ((3, carrier_index - 1), (4, carrier_index + 1)):
            if not 0 <= next_index < len(PHASE_CARRIERS): continue
            next_carrier = PHASE_CARRIERS[next_index]
            if next_carrier in pegs | bridges: continue
            child = next_carrier, pegs, bridges; child_cost = cost + 1
            if child_cost >= distance.get(child, 10 ** 9): continue
            distance[child] = child_cost; paths[child] = paths[state] + (("C", carrier, next_carrier, action),)
            heappush(queue, (child_cost, serial, child)); serial += 1
        occupied = pegs | bridges
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                if midpoint not in occupied or destination not in base_slots | {carrier} or destination in occupied:
                    continue
                child_pegs = set(pegs); child_bridges = set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination); child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = carrier, frozenset(child_pegs), frozenset(child_bridges); child_cost = cost + 2
                if child_cost >= distance.get(child, 10 ** 9): continue
                distance[child] = child_cost
                paths[child] = paths[state] + ((kind, source, destination),)
                heappush(queue, (child_cost, serial, child)); serial += 1
    return goals, len(distance)


def joint_solution(frame, max_states=1000000):
    phase_slots, phase_carrier, goals, phase_states = phase_one_goals(frame)
    local_slots = {(row, col - 20) for row, col in phase_slots if (row, col) != phase_carrier}
    slots = frozenset(WORLD_HOLES | WORLD_TARGET | WORLD_FAR_BRIDGES | local_slots)
    distance = {}; parent = {}; source_paths = {}; queue = []; serial = 0
    for phase_pegs, phase_bridges, phase_path in goals:
        world_pegs = set(WORLD_TARGET)
        for peg in phase_pegs:
            world_pegs.add(CARRIER_POSITIONS[0] if peg == phase_carrier else (peg[0], peg[1] - 20))
        world_bridges = set(WORLD_FAR_BRIDGES)
        world_bridges.update((row, col - 20) for row, col in phase_bridges)
        state = (0, 0, frozenset(world_pegs), frozenset(world_bridges))
        cost = 2 * len(phase_path)
        if cost >= distance.get(state, 10 ** 9): continue
        distance[state] = cost; parent[state] = None; source_paths[state] = phase_path
        heappush(queue, (cost, serial, state)); serial += 1

    goal_state = None
    while queue and len(distance) <= max_states:
        cost, _, state = heappop(queue)
        if cost != distance[state]: continue
        carrier_index, camera_index, pegs, bridges = state
        if len(pegs) == 1 and (36, 118) in pegs:
            goal_state = state; break
        if cost >= 102: continue
        carrier = CARRIER_POSITIONS[carrier_index]
        occupied = pegs | bridges
        for action, next_index in ((3, carrier_index - 1), (4, carrier_index + 1)):
            if not 0 <= next_index < len(CARRIER_POSITIONS): continue
            next_carrier = CARRIER_POSITIONS[next_index]
            child_pegs = set(pegs); child_bridges = set(bridges)
            if carrier in child_pegs:
                if next_carrier in occupied - {carrier}: continue
                child_pegs.remove(carrier); child_pegs.add(next_carrier)
            elif carrier in child_bridges:
                if next_carrier in occupied - {carrier}: continue
                child_bridges.remove(carrier); child_bridges.add(next_carrier)
            next_camera = camera_index + (next_index - carrier_index if carrier in pegs else 0)
            child = next_index, next_camera, frozenset(child_pegs), frozenset(child_bridges)
            child_cost = cost + 1
            if child_cost >= distance.get(child, 10 ** 9): continue
            distance[child] = child_cost; parent[child] = (state, (action,), ("C", carrier, next_carrier))
            heappush(queue, (child_cost, serial, child)); serial += 1

        scroll = 6 * camera_index
        destinations = slots | {carrier}
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            source_screen = source[1] - scroll
            if not 0 <= source_screen <= 60: continue
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                destination_screen = destination[1] - scroll
                if (
                    midpoint not in occupied | WORLD_FIXED
                    or destination not in destinations
                    or destination in occupied
                    or not 0 <= destination_screen <= 60
                ):
                    continue
                child_pegs = set(pegs); child_bridges = set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = carrier_index, camera_index, frozenset(child_pegs), frozenset(child_bridges)
                child_cost = cost + 2
                if child_cost >= distance.get(child, 10 ** 9): continue
                macro = (
                    (6, source_screen + 1, source[0] + 1),
                    (6, destination_screen + 1, destination[0] + 1),
                )
                distance[child] = child_cost; parent[child] = (state, macro, (kind, source, destination))
                heappush(queue, (child_cost, serial, child)); serial += 1

    if goal_state is None:
        return phase_states, len(goals), len(distance), None
    macros = []; descriptions = []; cursor = goal_state
    while parent[cursor] is not None:
        cursor, macro, description = parent[cursor]
        macros.append(macro); descriptions.append(description)
    macros.reverse(); descriptions.reverse()
    phase_path = source_paths[cursor]
    phase_actions = tuple(
        action
        for _, source, destination in phase_path
        for action in ((6, source[1] + 1, source[0] + 1), (6, destination[1] + 1, destination[0] + 1))
    )
    world_actions = tuple(action for macro in macros for action in macro)
    return phase_states, len(goals), len(distance), (phase_path, phase_actions + world_actions, descriptions, distance[goal_state])


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix: env.step(action)
    moving_goals, moving_states = moving_phase_goals(env.frame())
    compact_goals = []
    for carrier in PHASE_CARRIERS:
        candidates = [goal for goal in moving_goals if goal[1] == carrier]
        if candidates:
            best = min(candidates, key=lambda goal: goal[0])
            compact_goals.append((carrier, len(candidates), best[0], best[4]))
    print("MOVING_GOALS", moving_states, compact_goals)
    result = joint_solution(env.frame())
    print("GLOBAL_RESULT", result)
    if result[-1] is not None:
        _, actions, _, cost = result[-1]
        replay = env.clone()
        for action in actions:
            if isinstance(action, tuple): replay.step(*action)
            else: replay.step(action)
        print("GLOBAL_REPLAY", len(actions), cost, replay.levels_completed, actions)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
