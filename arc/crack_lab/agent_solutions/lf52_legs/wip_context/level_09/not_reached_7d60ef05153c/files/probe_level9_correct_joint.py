"""Joint level-9 search with correctly transported pre-entry carrier cargo."""

import json
from heapq import heappop, heappush

import gkm_try

from perception import safe_step
from probe_level9_global import (
    CARRIER_POSITIONS,
    PHASE_CARRIERS,
    WORLD_FAR_BRIDGES,
    WORLD_FIXED,
    WORLD_HOLES,
    WORLD_TARGET,
    pieces,
)


def phase_goals(frame):
    holes, root_pegs, root_bridges, carriers = pieces(frame)
    start_carrier = next(iter(carriers))
    slots = frozenset(holes | root_pegs | root_bridges)
    start = start_carrier, frozenset(root_pegs), frozenset(root_bridges)
    distance = {start: 0}; paths = {start: ()}
    queue = [(0, 0, start)]; serial = 1; goals = []; bridge_loaded = []
    while queue:
        cost, _, state = heappop(queue)
        if cost != distance[state]:
            continue
        carrier, pegs, bridges = state
        if carrier in pegs:
            goals.append((cost, carrier, pegs, bridges, paths[state])); continue
        if carrier in bridges:
            bridge_loaded.append((cost, carrier, pegs, bridges, paths[state]))

        carrier_index = PHASE_CARRIERS.index(carrier)
        for action, next_index in ((3, carrier_index - 1), (4, carrier_index + 1)):
            if not 0 <= next_index < len(PHASE_CARRIERS):
                continue
            next_carrier = PHASE_CARRIERS[next_index]
            occupied = pegs | bridges
            child_pegs, child_bridges = set(pegs), set(bridges)
            if carrier in child_pegs:
                if next_carrier in occupied - {carrier}:
                    continue
                child_pegs.remove(carrier); child_pegs.add(next_carrier)
            elif carrier in child_bridges:
                if next_carrier in occupied - {carrier}:
                    continue
                child_bridges.remove(carrier); child_bridges.add(next_carrier)
            child = next_carrier, frozenset(child_pegs), frozenset(child_bridges)
            child_cost = cost + 1
            if child_cost >= distance.get(child, 10 ** 9):
                continue
            distance[child] = child_cost
            paths[child] = paths[state] + (("C", carrier, next_carrier, action),)
            heappush(queue, (child_cost, serial, child)); serial += 1

        occupied = pegs | bridges
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                if (
                    midpoint not in occupied
                    or destination not in slots | {carrier}
                    or destination in occupied
                ):
                    continue
                child_pegs, child_bridges = set(pegs), set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = carrier, frozenset(child_pegs), frozenset(child_bridges)
                child_cost = cost + 2
                if child_cost >= distance.get(child, 10 ** 9):
                    continue
                distance[child] = child_cost
                paths[child] = paths[state] + ((kind, source, destination),)
                heappush(queue, (child_cost, serial, child)); serial += 1
    return slots, goals, bridge_loaded, len(distance)


def phase_actions(path):
    actions = []
    for item in path:
        if item[0] == "C":
            actions.append(item[3])
        else:
            _, source, destination = item
            actions.extend((
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            ))
    return tuple(actions)


def joint(frame, max_states=1000000):
    phase_slots, goals, bridge_loaded, phase_states = phase_goals(frame)
    local_slots = {(row, col - 20) for row, col in phase_slots}
    slots = frozenset(WORLD_HOLES | WORLD_TARGET | WORLD_FAR_BRIDGES | local_slots)
    distance = {}; parent = {}; source_paths = {}; queue = []; serial = 0
    for phase_cost, carrier, phase_pegs, phase_bridges, path in goals:
        carrier_index = PHASE_CARRIERS.index(carrier)
        world_pegs = set(WORLD_TARGET)
        world_pegs.update((row, col - 20) for row, col in phase_pegs)
        world_bridges = set(WORLD_FAR_BRIDGES)
        world_bridges.update((row, col - 20) for row, col in phase_bridges)
        state = carrier_index, 0, frozenset(world_pegs), frozenset(world_bridges)
        if phase_cost >= distance.get(state, 10 ** 9):
            continue
        distance[state] = phase_cost; parent[state] = None; source_paths[state] = path
        heappush(queue, (phase_cost, serial, state)); serial += 1

    goal_state = None
    while queue and len(distance) <= max_states:
        cost, _, state = heappop(queue)
        if cost != distance[state]:
            continue
        carrier_index, camera_index, pegs, bridges = state
        if len(pegs) == 1:
            goal_state = state; break
        if cost >= 102:
            continue
        carrier = CARRIER_POSITIONS[carrier_index]; occupied = pegs | bridges
        for action, next_index in ((3, carrier_index - 1), (4, carrier_index + 1)):
            if not 0 <= next_index < len(CARRIER_POSITIONS):
                continue
            next_carrier = CARRIER_POSITIONS[next_index]
            child_pegs, child_bridges = set(pegs), set(bridges)
            if carrier in child_pegs:
                if next_carrier in occupied - {carrier}:
                    continue
                child_pegs.remove(carrier); child_pegs.add(next_carrier)
            elif carrier in child_bridges:
                if next_carrier in occupied - {carrier}:
                    continue
                child_bridges.remove(carrier); child_bridges.add(next_carrier)
            next_camera = camera_index + (next_index - carrier_index if carrier in pegs else 0)
            child = next_index, next_camera, frozenset(child_pegs), frozenset(child_bridges)
            child_cost = cost + 1
            if child_cost >= distance.get(child, 10 ** 9):
                continue
            distance[child] = child_cost
            parent[child] = state, (action,), ("C", carrier, next_carrier)
            heappush(queue, (child_cost, serial, child)); serial += 1

        scroll = 6 * camera_index
        destinations = slots | {carrier}
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            source_screen = source[1] - scroll
            if not 0 <= source_screen <= 60:
                continue
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
                child_pegs, child_bridges = set(pegs), set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = carrier_index, camera_index, frozenset(child_pegs), frozenset(child_bridges)
                child_cost = cost + 2
                if child_cost >= distance.get(child, 10 ** 9):
                    continue
                macro = (
                    (6, source_screen + 1, source[0] + 1),
                    (6, destination_screen + 1, destination[0] + 1),
                )
                distance[child] = child_cost
                parent[child] = state, macro, (kind, source, destination)
                heappush(queue, (child_cost, serial, child)); serial += 1

    bridge_best = min(bridge_loaded, default=None, key=lambda item: item[0])
    if goal_state is None:
        return phase_states, len(goals), bridge_best, len(distance), None
    macros = []; descriptions = []; cursor = goal_state
    while parent[cursor] is not None:
        cursor, macro, description = parent[cursor]
        macros.append(macro); descriptions.append(description)
    macros.reverse(); descriptions.reverse()
    actions = phase_actions(source_paths[cursor]) + tuple(
        action for macro in macros for action in macro
    )
    return phase_states, len(goals), bridge_best, len(distance), (
        distance[goal_state], actions, descriptions, source_paths[cursor]
    )


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    result = joint(env.frame())
    print("L9_CORRECT_JOINT", result[:4], None if result[4] is None else (result[4][0], result[4][3]))
    if result[4] is None:
        return
    cost, actions, descriptions, _ = result[4]
    replay = env.clone()
    for action in actions:
        safe_step(replay, action)
    print("L9_CORRECT_REPLAY", cost, len(actions), int(replay.levels_completed), actions)
    print("L9_CORRECT_DESCRIPTIONS", descriptions)


gkm_try.A.run_program("lf52", probe)
