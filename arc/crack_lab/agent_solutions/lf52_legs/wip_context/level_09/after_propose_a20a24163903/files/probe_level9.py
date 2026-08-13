"""Compact, clean-room observations at the pristine level-9 entry."""

from collections import Counter, deque
from heapq import heappop, heappush
import json

import gkm_try
import numpy as np

from perception import (
    action_deltas,
    arr,
    color_counts,
    connected_components,
    frame_delta,
    safe_step,
)


def pieces(frame):
    blobs = connected_components(frame, colors=(1, 3, 9, 11, 12, 14, 15))
    border_positions = {
        (b.bbox[0] + 1, b.bbox[1] + 1)
        for b in blobs if b.color == 11 and b.area >= 4
    }
    carrier_positions = {
        b.top_left for b in blobs if b.color == 12 and b.size == (4, 4)
    } | border_positions
    return {
        "holes": tuple(b.top_left for b in blobs if b.color == 1 and b.size == (4, 4)),
        "pegs": tuple(b.top_left for b in blobs if b.color == 14 and b.size == (4, 4)),
        "carriers": tuple(sorted(carrier_positions)),
        "selected": tuple(b.bbox for b in blobs if b.color == 3 and b.area >= 4),
        "c9": tuple((b.bbox, b.area) for b in blobs if b.color == 9 and b.area >= 4),
        "borders": tuple((b.bbox, b.area) for b in blobs if b.color == 11 and b.area >= 4),
        "bridges": tuple(b.top_left for b in blobs if b.color == 9 and b.size == (4, 4)),
        "fixed_bridges": tuple(
            (b.bbox[0] + 1, b.bbox[1])
            for b in blobs if b.color == 15 and b.size == (4, 4)
        ),
    }


def transition(before, after):
    a, b = arr(before), arr(after)
    changed = a != b
    changed[0, :] = False
    pairs = Counter(zip(a[changed].tolist(), b[changed].tolist()))
    return frame_delta(a[1:, :], b[1:, :])["count"], tuple(sorted(pairs.items()))


def key_effects(node):
    base = arr(node.frame())
    effects = []
    for action in (1, 2, 3, 4):
        child = node.clone()
        safe_step(child, action)
        count = int(np.count_nonzero(base[1:, :] != arr(child.frame())[1:, :]))
        if count:
            effects.append((action, count))
    return tuple(effects)


def legal_peg_moves(node):
    state = pieces(node.frame())
    destinations = set(state["holes"]) | set(state["carriers"])
    moves = []
    for source in state["pegs"]:
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = (source[0] + dr, source[1] + dc)
            if destination not in destinations:
                continue
            child = node.clone()
            safe_step(child, (6, source[1] + 1, source[0] + 1))
            safe_step(child, (6, destination[1] + 1, destination[0] + 1))
            after = pieces(child.frame())
            if after["pegs"] != state["pegs"] or child.levels_completed != node.levels_completed:
                moves.append((source, destination, after["pegs"], child.levels_completed))
    return tuple(moves)


def legal_piece_moves(node):
    state = pieces(node.frame())
    sources = tuple(("peg", p) for p in state["pegs"])
    sources += tuple(("bridge", p) for p in state["bridges"])
    destinations = set(state["holes"]) | set(state["carriers"])
    before_key = (state["pegs"], state["bridges"])
    moves = []
    for kind, source in sources:
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = (source[0] + dr, source[1] + dc)
            if destination not in destinations:
                continue
            child = node.clone()
            safe_step(child, (6, source[1] + 1, source[0] + 1))
            safe_step(child, (6, destination[1] + 1, destination[0] + 1))
            after = pieces(child.frame())
            after_key = (after["pegs"], after["bridges"])
            if after_key != before_key or child.levels_completed != node.levels_completed:
                moves.append((kind, source, destination, after_key, child.levels_completed))
    return tuple(moves)


def symbolic_solution(frame, max_states=250000):
    state = pieces(frame)
    slots = frozenset(
        set(state["holes"]) | set(state["pegs"]) |
        set(state["bridges"]) | set(state["carriers"])
    )
    carrier = next(iter(state["carriers"]))
    start = (frozenset(state["pegs"]), frozenset(state["bridges"]))
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (pegs, bridges), path = queue.popleft()
        if len(pegs) == 1 and carrier in pegs:
            return path, len(seen)
        occupied = pegs | bridges
        sources = tuple(("peg", source) for source in sorted(pegs))
        sources += tuple(("bridge", source) for source in sorted(bridges))
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in slots
                    or destination in occupied
                ):
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
                child = (frozenset(child_pegs), frozenset(child_bridges))
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, path + ((kind, source, destination),)))
    return None, len(seen)


def symbolic_carrier_entry(frame, kind, max_states=50000):
    state = pieces(frame)
    slots = frozenset(
        set(state["holes"]) | set(state["pegs"]) |
        set(state["bridges"]) | set(state["carriers"])
    )
    carrier = next(iter(state["carriers"]))
    start = (frozenset(state["pegs"]), frozenset(state["bridges"]))
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (pegs, bridges), path = queue.popleft()
        occupants = pegs if kind == "peg" else bridges
        if carrier in occupants:
            return path, len(seen)
        occupied = pegs | bridges
        sources = tuple(("peg", source) for source in sorted(pegs))
        sources += tuple(("bridge", source) for source in sorted(bridges))
        for mover, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if midpoint not in occupied or destination not in slots or destination in occupied:
                    continue
                child_pegs = set(pegs)
                child_bridges = set(bridges)
                if mover == "peg":
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
                queue.append((child, path + ((mover, source, destination),)))
    return None, len(seen)


def world_symbolic_solution(
        holes, initial_bridges, fixed_bridges, target_pegs, carrier_positions,
        max_states=500000, max_cost=120):
    carrier_positions = tuple(sorted(carrier_positions, key=lambda p: p[1]))
    slots = frozenset(set(holes) | set(initial_bridges) | set(target_pegs))
    start_carrier = carrier_positions[0]
    start = (
        0,
        0,
        frozenset(set(target_pegs) | {start_carrier}),
        frozenset(initial_bridges),
    )
    distance = {start: 0}
    parent = {start: None}
    serial = 0
    queue = [(0, serial, start)]
    goal = None
    while queue and len(distance) <= max_states:
        cost, _, state = heappop(queue)
        if cost != distance[state]:
            continue
        carrier_index, camera_index, pegs, bridges = state
        carrier = carrier_positions[carrier_index]
        if len(pegs) == 1 and carrier in pegs:
            goal = state
            break
        if cost >= max_cost:
            continue
        occupied = pegs | bridges

        for action, next_index in ((3, carrier_index - 1), (4, carrier_index + 1)):
            if not 0 <= next_index < len(carrier_positions):
                continue
            next_carrier = carrier_positions[next_index]
            child_pegs = set(pegs)
            child_bridges = set(bridges)
            if carrier in child_pegs:
                if next_carrier in occupied - {carrier}:
                    continue
                child_pegs.remove(carrier)
                child_pegs.add(next_carrier)
            elif carrier in child_bridges:
                if next_carrier in occupied - {carrier}:
                    continue
                child_bridges.remove(carrier)
                child_bridges.add(next_carrier)
            next_camera_index = camera_index
            if carrier in pegs:
                next_camera_index += next_index - carrier_index
            child = (
                next_index, next_camera_index,
                frozenset(child_pegs), frozenset(child_bridges),
            )
            child_cost = cost + 1
            if child_cost >= distance.get(child, 10 ** 9):
                continue
            distance[child] = child_cost
            parent[child] = (state, (action,), ("carrier", carrier, next_carrier))
            serial += 1
            heappush(queue, (child_cost, serial, child))

        destinations = slots | {carrier}
        sources = tuple(("peg", source) for source in sorted(pegs))
        sources += tuple(("bridge", source) for source in sorted(bridges))
        scroll = 6 * camera_index
        for kind, source in sources:
            source_screen_col = source[1] - scroll
            if not 0 <= source_screen_col <= 60:
                continue
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                destination_screen_col = destination[1] - scroll
                if (
                    midpoint not in occupied | fixed_bridges
                    or destination not in destinations
                    or destination in occupied
                    or not 0 <= destination_screen_col <= 60
                ):
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
                child = (
                    carrier_index,
                    camera_index,
                    frozenset(child_pegs),
                    frozenset(child_bridges),
                )
                child_cost = cost + 2
                if child_cost >= distance.get(child, 10 ** 9):
                    continue
                macro = (
                    (6, source_screen_col + 1, source[0] + 1),
                    (6, destination_screen_col + 1, destination[0] + 1),
                )
                distance[child] = child_cost
                parent[child] = (state, macro, (kind, source, destination))
                serial += 1
                heappush(queue, (child_cost, serial, child))

    if goal is None:
        diagnostic_candidates = [
            state for state in distance
            if (
                (24, 52) in state[2]
                and carrier_positions[state[0]] not in state[2]
                and state[0] == 5
            )
        ]
        if not diagnostic_candidates:
            diagnostic_candidates = [
                state for state in distance if (24, 52) in state[2]
            ]
        diagnostic_actions = None
        diagnostic_moves = None
        diagnostic_state = None
        if diagnostic_candidates:
            diagnostic_state = min(diagnostic_candidates, key=distance.get)
            trace_actions = []
            trace_moves = []
            cursor = diagnostic_state
            while parent[cursor] is not None:
                cursor, macro, description = parent[cursor]
                trace_actions.append(macro)
                trace_moves.append(description)
            trace_actions.reverse()
            trace_moves.reverse()
            diagnostic_actions = tuple(
                action for macro in trace_actions for action in macro
            )
            diagnostic_moves = tuple(trace_moves)
        single_candidates = [state for state in distance if len(state[2]) == 1]
        single_actions = None
        single_moves = None
        single_state = None
        if single_candidates:
            single_state = min(single_candidates, key=distance.get)
            trace_actions = []
            trace_moves = []
            cursor = single_state
            while parent[cursor] is not None:
                cursor, macro, description = parent[cursor]
                trace_actions.append(macro)
                trace_moves.append(description)
            trace_actions.reverse()
            trace_moves.reverse()
            single_actions = tuple(
                action for macro in trace_actions for action in macro
            )
            single_moves = tuple(trace_moves)
        exit_candidates = [state for state in distance if (36, 118) in state[2]]
        exit_actions = None
        exit_moves = None
        exit_state = None
        if exit_candidates:
            exit_state = min(exit_candidates, key=distance.get)
            trace_actions = []
            trace_moves = []
            cursor = exit_state
            while parent[cursor] is not None:
                cursor, macro, description = parent[cursor]
                trace_actions.append(macro)
                trace_moves.append(description)
            trace_actions.reverse()
            trace_moves.reverse()
            exit_actions = tuple(action for macro in trace_actions for action in macro)
            exit_moves = tuple(trace_moves)
        single_positions = sorted({
            next(iter(pegs))
            for _, _, pegs, _ in distance
            if len(pegs) == 1
        })
        peg_positions = sorted({
            peg for _, _, pegs, _ in distance for peg in pegs
        })
        pair_distances = [
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for _, _, pegs, _ in distance
            if len(pegs) == 2
            for a, b in (tuple(sorted(pegs)),)
        ]
        diagnostics = {
            "single_positions": tuple(single_positions),
            "peg_positions": tuple(peg_positions),
            "min_pair_distance": min(pair_distances) if pair_distances else None,
            "target24_actions": diagnostic_actions,
            "target24_moves": diagnostic_moves,
            "target24_state": diagnostic_state,
            "single_actions": single_actions,
            "single_moves": single_moves,
            "single_state": single_state,
            "exit_actions": exit_actions,
            "exit_moves": exit_moves,
            "exit_state": exit_state,
        }
        return None, diagnostics, len(distance)
    macros = []
    descriptions = []
    cursor = goal
    while parent[cursor] is not None:
        cursor, macro, description = parent[cursor]
        macros.append(macro)
        descriptions.append(description)
    macros.reverse()
    descriptions.reverse()
    actions = tuple(action for macro in macros for action in macro)
    return actions, tuple(descriptions), len(distance)


def dynamic_solution(root, max_states=4000, max_cost=60):
    base_level = root.levels_completed

    def state_key(node):
        return arr(node.frame())[1:, :].tobytes()

    def metric(node):
        state = pieces(node.frame())
        pegs = state["pegs"]
        carrier = state["carriers"][0] if state["carriers"] else None
        pair_distance = (
            min(
                abs(a[0] - b[0]) + abs(a[1] - b[1])
                for i, a in enumerate(pegs) for b in pegs[i + 1:]
            ) if len(pegs) >= 2 else 0
        )
        carrier_distance = (
            min(abs(p[0] - carrier[0]) + abs(p[1] - carrier[1]) for p in pegs)
            if pegs and carrier else 999
        )
        return len(pegs), pair_distance, carrier_distance

    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best_cost = {state_key(root): 0}
    best_metric = metric(root)
    best_path = ()
    while queue and len(best_cost) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best_cost.get(state_key(node)):
            continue
        if node.levels_completed > base_level:
            return path, len(best_cost), best_metric, best_path
        if cost >= max_cost:
            continue

        children = []
        before_frame = arr(node.frame())
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            if (
                child.levels_completed > base_level
                or np.any(before_frame[1:, :] != arr(child.frame())[1:, :])
            ):
                children.append(((action,), child))

        state = pieces(node.frame())
        occupied = set(state["pegs"]) | set(state["bridges"])
        destinations = set(state["holes"]) | set(state["carriers"])
        sources = tuple(state["pegs"]) + tuple(state["bridges"])
        seen_macros = set()
        for source in sources:
            for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                destination = (source[0] + dr, source[1] + dc)
                if destination not in destinations or destination in occupied:
                    continue
                macro = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                if macro in seen_macros:
                    continue
                seen_macros.add(macro)
                child = node.clone()
                for action in macro:
                    safe_step(child, action)
                after = pieces(child.frame())
                moved = (
                    after["pegs"] != state["pegs"]
                    or after["bridges"] != state["bridges"]
                )
                if child.levels_completed > base_level or (moved and not after["selected"]):
                    children.append((macro, child))

        for macro, child in children:
            child_path = path + macro
            child_cost = cost + len(macro)
            key = state_key(child)
            if child_cost >= best_cost.get(key, 10 ** 9):
                continue
            best_cost[key] = child_cost
            child_metric = metric(child)
            if child_metric < best_metric:
                best_metric = child_metric
                best_path = child_path
            serial += 1
            heappush(queue, (child_cost, serial, child, child_path))
    return None, len(best_cost), best_metric, best_path


def summarize(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        env.step(action)
    frame = env.frame()
    blobs = [
        (blob.color, blob.bbox, blob.area, blob.size)
        for blob in connected_components(frame, min_area=4)
    ]
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env).items()
    }
    print("ENTRY", env.levels_completed, tuple(env.actions), frame.shape)
    print("COLORS", color_counts(frame))
    print("BLOBS", blobs)
    print("DELTAS", deltas)
    for top, left in ((42, 18), (48, 42), (42, 24), (42, 42), (36, 42)):
        print("TILE", (top, left), arr(frame)[top:top + 4, left:left + 4].tolist())

    p2_clicks = []
    for row, col in zip(*np.where(arr(frame) == 14)):
        if row < 48:
            continue
        node = env.clone()
        safe_step(node, (6, int(col), int(row)))
        delta = transition(frame, node.frame())
        if delta[0]:
            p2_clicks.append(((int(col), int(row)), delta))
    print("P2_SELECTABLE_PIXELS", p2_clicks)

    moved = env.clone()
    safe_step(moved, (6, 19, 43))
    safe_step(moved, (6, 31, 43))
    print("LEGAL_ROOT", legal_peg_moves(env))
    print("LEGAL_AFTER_FIRST", legal_peg_moves(moved))
    print("PIECES_ROOT", legal_piece_moves(env))
    print("PIECES_AFTER_FIRST", legal_piece_moves(moved))

    bridge_moved = moved.clone()
    for action in ((6, 25, 43), (6, 37, 43)):
        safe_step(bridge_moved, action)
    print("AFTER_BRIDGE_MOVE", transition(frame, bridge_moved.frame()), pieces(bridge_moved.frame()))
    print("PIECES_AFTER_BRIDGE", legal_piece_moves(bridge_moved))

    solution, searched = symbolic_solution(frame)
    print("SYMBOLIC", searched, solution)
    if solution is not None:
        pre_entry = env.clone()
        for _, source, destination in solution[:-1]:
            safe_step(pre_entry, (6, source[1] + 1, source[0] + 1))
            safe_step(pre_entry, (6, destination[1] + 1, destination[0] + 1))
        bypasses = []
        for count in (1, 2, 3):
            bypass = pre_entry.clone()
            for _ in range(count):
                safe_step(bypass, 4)
            before_jump = arr(bypass.frame())[1:, :].copy()
            safe_step(bypass, (6, 31, 37))
            safe_step(bypass, (6, 43, 37))
            bypasses.append((
                count,
                int(np.count_nonzero(before_jump != arr(bypass.frame())[1:, :])),
                bypass.levels_completed,
                pieces(bypass.frame()),
            ))
        print("CARRIER_BYPASS", bypasses)
    for carrier_kind in ("peg", "bridge"):
        entry_solution, entry_searched = symbolic_carrier_entry(frame, carrier_kind)
        entry_replay = env.clone()
        if entry_solution is not None:
            for _, source, destination in entry_solution:
                safe_step(entry_replay, (6, source[1] + 1, source[0] + 1))
                safe_step(entry_replay, (6, destination[1] + 1, destination[0] + 1))
        print(
            "CARRIER_ENTRY", carrier_kind, entry_searched, entry_solution,
            entry_replay.levels_completed, pieces(entry_replay.frame()),
        )
    if solution is not None:
        verified = env.clone()
        progress = []
        for kind, source, destination in solution:
            before_state = pieces(verified.frame())
            safe_step(verified, (6, source[1] + 1, source[0] + 1))
            safe_step(verified, (6, destination[1] + 1, destination[0] + 1))
            after_state = pieces(verified.frame())
            progress.append((
                kind, source, destination,
                len(after_state["pegs"]), after_state["pegs"],
                after_state["bridges"], verified.levels_completed,
                before_state != after_state,
            ))
        print("SYMBOLIC_REPLAY", len(solution) * 2, verified.levels_completed, progress)
        phase_deltas = {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(verified).items()
        }
        print("PHASE1_STATE", color_counts(verified.frame()), pieces(verified.frame()))
        print("PHASE1_DELTAS", phase_deltas)
        print("PHASE1_LEGAL", legal_piece_moves(verified))
        phase2_solution, phase2_searched = symbolic_solution(verified.frame())
        print("PHASE2_SYMBOLIC", phase2_searched, phase2_solution)
        transport = []
        for count in range(1, 17):
            node = verified.clone()
            for _ in range(count):
                safe_step(node, 4)
            state = pieces(node.frame())
            local_solution, local_searched = symbolic_solution(node.frame())
            transport.append((
                count, node.levels_completed, state["pegs"],
                state["carriers"], state["bridges"], len(state["holes"]),
                key_effects(node), legal_piece_moves(node),
                local_searched, local_solution,
            ))
        print("PHASE1_TRANSPORT", transport)

        world_holes = set()
        world_bridges = set()
        world_fixed_bridges = set()
        world_pegs = set()
        carrier_positions = set()
        for scroll in range(15):
            node = verified.clone()
            for _ in range(scroll):
                safe_step(node, 4)
            state = pieces(node.frame())
            carrier_screen = next(iter(state["carriers"]))
            carrier_world = (carrier_screen[0], carrier_screen[1] + 6 * scroll)
            carrier_positions.add(carrier_world)
            world_holes.update((r, c + 6 * scroll) for r, c in state["holes"])
            world_bridges.update((r, c + 6 * scroll) for r, c in state["bridges"])
            world_fixed_bridges.update(
                (r, c + 6 * scroll) for r, c in state["fixed_bridges"]
            )
            world_pegs.update(
                (r, c + 6 * scroll)
                for r, c in state["pegs"]
                if (r, c) != carrier_screen
            )
        print("WORLD_HOLES", tuple(sorted(world_holes)))
        print("WORLD_BRIDGES", tuple(sorted(world_bridges)))
        print("WORLD_FIXED", tuple(sorted(world_fixed_bridges)))
        print("WORLD_PEGS", tuple(sorted(world_pegs)))
        print("WORLD_CARRIER", tuple(sorted(carrier_positions)))
        world_actions, world_moves, world_searched = world_symbolic_solution(
            world_holes,
            world_bridges,
            world_fixed_bridges,
            world_pegs,
            carrier_positions,
        )
        print("WORLD_SOLUTION", world_searched, world_actions, world_moves)
        if world_actions is not None:
            world_replay = verified.clone()
            for action in world_actions:
                safe_step(world_replay, action)
            print(
                "WORLD_REPLAY", len(world_actions), world_replay.levels_completed,
                color_counts(world_replay.frame()), pieces(world_replay.frame()),
            )
        elif world_moves.get("exit_actions") is not None:
            exit_replay = verified.clone()
            for action in world_moves["exit_actions"]:
                safe_step(exit_replay, action)
            print(
                "EXIT_REPLAY", len(world_moves["exit_actions"]),
                exit_replay.levels_completed, pieces(exit_replay.frame()),
                world_moves["exit_state"],
            )
        elif world_moves.get("single_actions") is not None:
            print("SINGLE_ACTIONS", world_moves["single_actions"])
            single_replay = verified.clone()
            for action in world_moves["single_actions"]:
                safe_step(single_replay, action)
            print(
                "SINGLE_REPLAY", len(world_moves["single_actions"]),
                single_replay.levels_completed, pieces(single_replay.frame()),
                world_moves["single_moves"],
            )
        elif world_moves.get("target24_actions") is not None:
            target_replay = verified.clone()
            action_iter = iter(world_moves["target24_actions"])
            trace = []
            scroll_index = 0
            for description in world_moves["target24_moves"]:
                macro_size = 1 if description[0] == "carrier" else 2
                macro = tuple(next(action_iter) for _ in range(macro_size))
                before_macro = arr(target_replay.frame()).copy()
                for action in macro:
                    safe_step(target_replay, action)
                    if action == 4:
                        scroll_index += 1
                    elif action == 3:
                        scroll_index -= 1
                actual = pieces(target_replay.frame())
                trace.append((
                    description, macro,
                    int(np.count_nonzero(before_macro[1:, :] != arr(target_replay.frame())[1:, :])),
                    tuple((r, c + 6 * scroll_index) for r, c in actual["pegs"]),
                    tuple((r, c + 6 * scroll_index) for r, c in actual["bridges"]),
                ))
            print(
                "TARGET24_REPLAY", len(world_moves["target24_actions"]),
                target_replay.levels_completed, pieces(target_replay.frame()),
                key_effects(target_replay), legal_piece_moves(target_replay),
            )
            print("TARGET24_TRACE", trace)
            loaded_actions = world_moves["target24_actions"][2:]
            loaded_replay = verified.clone()
            for action in loaded_actions:
                safe_step(loaded_replay, action)
            print(
                "LOADED_TARGET24", len(loaded_actions),
                loaded_replay.levels_completed, pieces(loaded_replay.frame()),
                key_effects(loaded_replay), legal_piece_moves(loaded_replay),
            )
            dynamic_actions, dynamic_searched, dynamic_metric, dense_path = dynamic_solution(
                loaded_replay, max_states=0,
            )
            print(
                "DYNAMIC_SOLUTION", dynamic_searched, dynamic_metric,
                dynamic_actions, dense_path,
            )
            if dynamic_actions is not None:
                dynamic_replay = loaded_replay.clone()
                for action in dynamic_actions:
                    safe_step(dynamic_replay, action)
                print(
                    "DYNAMIC_REPLAY", len(dynamic_actions),
                    dynamic_replay.levels_completed, pieces(dynamic_replay.frame()),
                )
            completion_macros = (
                (4, 4, 4),
                ((6, 23, 37), (6, 23, 25)),
                ((6, 5, 25), (6, 17, 25)),
                ((6, 17, 25), (6, 29, 25)),
                (4,),
                ((6, 29, 25), (6, 29, 37)),
            )
            completion_replay = loaded_replay.clone()
            completion_trace = []
            for macro in completion_macros:
                before_macro = arr(completion_replay.frame()).copy()
                for action in macro:
                    safe_step(completion_replay, action)
                completion_trace.append((
                    macro,
                    int(np.count_nonzero(before_macro[1:, :] != arr(completion_replay.frame())[1:, :])),
                    completion_replay.levels_completed,
                    pieces(completion_replay.frame()),
                ))
            print("COMPLETION_TRACE", completion_trace)

    contextual = (
        (7, (6, 43, 49)),
        (7, (6, 43, 49), (6, 43, 37)),
        ((6, 19, 43), 7),
        ((6, 19, 43), (6, 31, 43), 7),
        ((6, 19, 43), 1),
        ((6, 19, 43), 2),
        ((6, 19, 43), 3),
        ((6, 19, 43), 4),
        ((6, 25, 43), 1),
        ((6, 25, 43), 2),
        ((6, 25, 43), 3),
        ((6, 25, 43), 4),
        ((6, 25, 49), 1),
        ((6, 25, 49), 1, 1),
    )
    for path in contextual:
        node = env.clone()
        for action in path:
            safe_step(node, action)
        print("CONTEXT", path, transition(frame, node.frame()), pieces(node.frame()))

    offscreen_reentry = []
    for vertical_action in (1, 2):
        for count in range(1, 9):
            path = (4,) * 4 + (vertical_action,) * count + (3,)
            node = env.clone()
            for action in path:
                safe_step(node, action)
            p = pieces(node.frame())
            offscreen_reentry.append((vertical_action, count, p["carriers"], p["borders"]))
    print("OFFSCREEN_REENTRY", offscreen_reentry)

    probes = (
        (4,), (4, 4), (4, 4, 4), (4, 1),
        ((6, 19, 43),),       # first peg
        ((6, 43, 49),),       # second peg
        ((6, 43, 37),),       # carrier
        ((6, 31, 19),),       # color-9 lattice object
        ((6, 19, 43), (6, 31, 43)),  # peg over color-9 object
    )
    for path in probes:
        node = env.clone()
        for action in path:
            safe_step(node, action)
        print("PROBE", path, transition(frame, node.frame()), pieces(node.frame()))

    queue = deque([(env.clone(), ())])
    seen = {arr(frame)[1:, :].tobytes()}
    key_states = [((), pieces(frame)["carriers"], pieces(frame)["borders"])]
    while queue and len(seen) < 50:
        node, path = queue.popleft()
        if len(path) >= 8:
            continue
        for action in (1, 2, 3, 4, 7):
            child = node.clone()
            safe_step(child, action)
            key = arr(child.frame())[1:, :].tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_pieces = pieces(child.frame())
            key_states.append((path + (action,), child_pieces["carriers"], child_pieces["borders"]))
            queue.append((child, path + (action,)))
    print("KEY_STATES", len(seen), key_states)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", summarize)
