"""Search all shortest first-relay arrangements before extending level 9."""

import json
import os
import sys
from collections import deque
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


def initial_board(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    cells = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 9, 12, 14)
    )
    pegs = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    )
    bridges = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    )
    carriers = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    )
    return cells, pegs, bridges, carriers


def model_children(cells, state):
    pegs, bridges = state
    occupied = pegs | bridges
    for kind, source in (
        tuple(("peg", cell) for cell in sorted(pegs))
        + tuple(("bridge", cell) for cell in sorted(bridges))
    ):
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                midpoint not in occupied
                or destination not in cells
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
            yield (
                frozenset(child_pegs), frozenset(child_bridges)
            ), (source, destination)


def shortest_relay_paths(frame, max_depth=16):
    cells, pegs, bridges, carriers = initial_board(frame)
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen_depth = {start: 0}
    goals = []
    goal_depth = None
    while queue:
        state, path = queue.popleft()
        depth = len(path)
        if len(state[0]) == 1 and state[0] <= carriers:
            if goal_depth is None:
                goal_depth = depth
            goals.append((state, path))
            continue
        if depth >= max_depth or (
            goal_depth is not None and depth >= goal_depth
        ):
            continue
        for child, move in model_children(cells, state):
            child_depth = depth + 1
            known = seen_depth.get(child)
            if known is not None:
                continue
            seen_depth[child] = child_depth
            queue.append((child, path + (move,)))
    unique = {}
    for state, path in goals:
        unique.setdefault(state, path)
    return tuple(sorted({len(path) for path in unique.values()})), tuple(unique.values()), len(seen_depth)


def shortest_relay_paths_astar(frame, bound=14, max_states=300000,
                               max_goals=20):
    """Enumerate only layouts that can match the known shortest load cost."""
    cells, pegs, bridges, carriers = initial_board(frame)
    carrier = next(iter(carriers))
    start = (pegs, bridges)

    def lower_bound(state):
        state_pegs = state[0]
        distance = min(
            (abs(peg[0] - carrier[0]) + abs(peg[1] - carrier[1])) // 12
            for peg in state_pegs
        )
        return max(1, distance) if len(state_pegs) > 1 else distance

    serial = 0
    queue = [(3 * lower_bound(start), 0, serial, start, ())]
    seen_depth = {start: 0}
    goals = {}
    while queue and len(seen_depth) <= max_states:
        estimate, depth, _, state, path = heappop(queue)
        if depth != seen_depth.get(state):
            continue
        if len(state[0]) == 1 and state[0] <= carriers:
            goals.setdefault(state, path)
            if len(goals) >= max_goals:
                break
            continue
        if depth >= bound:
            continue
        for child, move in model_children(cells, state):
            child_depth = depth + 1
            if child_depth + lower_bound(child) > bound:
                continue
            if child_depth >= seen_depth.get(child, 10 ** 9):
                continue
            seen_depth[child] = child_depth
            serial += 1
            heappush(queue, (
                child_depth + 3 * lower_bound(child), child_depth, serial,
                child, path + (move,),
            ))
    unique = tuple(goals.values())
    return ((bound,) if unique else ()), unique, len(seen_depth)


def play_moves(env, moves):
    for source, destination in moves:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def board(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    return (
        frozenset(holes | pegs | bridges | carriers | fixed),
        frozenset(pegs), frozenset(bridges),
        frozenset(carriers), frozenset(fixed),
    )


def compact(frame):
    _, pegs, bridges, carriers, fixed = board(frame)
    return (
        tuple(sorted(pegs)), tuple(sorted(bridges)),
        tuple(sorted(carriers)), tuple(sorted(fixed)),
    )


def click_macros(frame):
    cells, pegs, bridges, _, fixed = board(frame)
    occupied = pegs | bridges | fixed
    out = []
    for source in sorted(pegs | bridges):
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                midpoint in occupied
                and destination in cells
                and destination not in occupied
            ):
                out.append((
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ))
    return tuple(out)


def frame_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def extend(roots, max_total_cost=56, max_states=12000):
    serial = 0
    queue = []
    best = {}
    for node, prefix in roots:
        cost = len(prefix)
        node_key = frame_key(node)
        if cost >= best.get(node_key, 10 ** 9):
            continue
        best[node_key] = cost
        serial += 1
        heappush(queue, (cost, serial, node, prefix))
    base_level = int(roots[0][0].levels_completed)
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(frame_key(node)) or cost >= max_total_cost:
            continue
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += click_macros(node.frame())
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > max_total_cost:
                continue
            child = node.clone()
            for action in macro:
                safe_step(child, action)
                if child.levels_completed > base_level:
                    return path + macro, len(best), child_cost, child
            child_key = frame_key(child)
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, path + macro))
        if len(best) % 1000 == 0:
            print("extend_progress", len(best), cost, flush=True)
    return None, len(best), None, None


def extend_replay(roots, max_total_cost=120, max_states=6000):
    def reconstruct(root_index, suffix):
        node = roots[root_index][0].clone()
        for action in suffix:
            safe_step(node, action)
        return node

    serial = 0
    queue = []
    best = {}
    for root_index, (node, prefix) in enumerate(roots):
        cost = len(prefix)
        node_key = frame_key(node)
        if cost >= best.get(node_key, 10 ** 9):
            continue
        best[node_key] = cost
        serial += 1
        heappush(queue, (cost, serial, root_index, ()))
    base_level = int(roots[0][0].levels_completed)
    processed = 0
    while queue and len(best) <= max_states:
        cost, _, root_index, suffix = heappop(queue)
        node = reconstruct(root_index, suffix)
        if cost != best.get(frame_key(node)) or cost >= max_total_cost:
            continue
        processed += 1
        if processed % 250 == 0:
            print("replay_progress", processed, len(best), cost, flush=True)
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += click_macros(node.frame())
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > max_total_cost:
                continue
            child_suffix = suffix + macro
            child = reconstruct(root_index, child_suffix)
            if child.levels_completed > base_level:
                return (
                    roots[root_index][1] + child_suffix,
                    len(best), child_cost, child,
                )
            child_key = frame_key(child)
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(
                queue,
                (child_cost, serial, root_index, child_suffix),
            )
    return None, len(best), None, None


def next_frontiers(roots, max_total_cost, max_states=6000):
    serial = 0
    queue = []
    best = {}
    for node, prefix in roots:
        cost = len(prefix)
        node_key = frame_key(node)
        if cost >= best.get(node_key, 10 ** 9):
            continue
        best[node_key] = cost
        serial += 1
        heappush(queue, (cost, serial, node, prefix))
    base_level = int(roots[0][0].levels_completed)
    events = {}
    event_floor = None
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if event_floor is not None and cost > event_floor + 2:
            break
        if cost != best.get(frame_key(node)) or cost >= max_total_cost:
            continue
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += click_macros(node.frame())
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > max_total_cost:
                continue
            before = arr(node.frame()).copy()
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            child_path = path + macro
            if child.levels_completed > base_level:
                return ((child, child_path),), len(best), True
            delta = frame_delta(before, child.frame())
            if len(macro) == 2 and delta["count"] > 500:
                if event_floor is None or child_cost < event_floor:
                    event_floor = child_cost
                child_key = frame_key(child)
                old = events.get(child_key)
                if old is None or child_cost < len(old[1]):
                    events[child_key] = (child, child_path)
                continue
            child_key = frame_key(child)
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, child_path))
    ordered = tuple(sorted(events.values(), key=lambda item: len(item[1])))
    return ordered, len(best), False


def reload_paths(frame, max_depth=24):
    cells, pegs, bridges, _, fixed = board(frame)
    borders = [
        blob for blob in connected_components(frame, colors=(11,))
        if blob.area == 20 and blob.size == (6, 6)
    ]
    if len(borders) != 1:
        return (), 0
    carrier = (borders[0].bbox[0] + 1, borders[0].bbox[1] + 1)
    static_cells = frozenset(cells - {carrier})
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    goals = {}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        state_pegs, state_bridges = state
        occupied = state_pegs | state_bridges | fixed
        destinations = static_cells | {carrier}
        for kind, source in (
            tuple(("peg", cell) for cell in sorted(state_pegs))
            + tuple(("bridge", cell) for cell in sorted(state_bridges))
        ):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
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
                actions = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                child_path = path + actions
                if destination == carrier and carrier not in state_pegs | state_bridges:
                    goals.setdefault((kind, child), child_path)
                    continue
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, child_path))
    ordered = tuple(sorted(
        ((len(path), kind, state, path)
         for (kind, state), path in goals.items()),
        key=lambda item: (item[0], item[1], item[2]),
    ))
    return ordered, len(seen)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_level = int(env.levels_completed)

    if os.environ.get("L9_FAST") == "1":
        depth, relay_paths, model_states = shortest_relay_paths_astar(
            env.frame(), max_states=int(os.environ.get("L9_MODEL_STATES", "300000"))
        )
    else:
        depth, relay_paths, model_states = shortest_relay_paths(env.frame())
    print("relay_models", depth, len(relay_paths), model_states, flush=True)
    roots_by_key = {}
    for index, moves in enumerate(relay_paths):
        clone = env.clone()
        play_moves(clone, moves)
        if clone.levels_completed != base_level:
            print("early_reward", index, flush=True)
        clone_key = frame_key(clone)
        actions = tuple(
            action
            for source, destination in moves
            for action in (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
        )
        roots_by_key.setdefault(clone_key, (clone, actions, moves))
    roots = tuple((clone, actions) for clone, actions, _ in roots_by_key.values())
    print("relay_roots", len(roots), [compact(root.frame()) for root, _ in roots],
          flush=True)
    if os.environ.get("L9_MODE") == "frontiers":
        layer_roots = roots
        for layer, limit in ((2, 70), (3, 100), (4, 130)):
            layer_roots, explored, won = next_frontiers(layer_roots, limit)
            print("frontier_layer", layer, explored, won, len(layer_roots),
                  [(len(path), compact(node.frame()))
                   for node, path in layer_roots], flush=True)
            if won or not layer_roots:
                break
    elif os.environ.get("L9_MODE") == "reloads":
        events = {}
        for root_index, (root, prefix) in enumerate(roots):
            models, states = reload_paths(root.frame())
            print("reload_models", root_index, states,
                  [(cost, kind) for cost, kind, _, _ in models], flush=True)
            for extra_cost, kind, _, suffix in models:
                clone = root.clone()
                before = arr(clone.frame()).copy()
                for action in suffix:
                    safe_step(clone, action)
                delta = frame_delta(before, clone.frame())["count"]
                if delta <= 500:
                    continue
                path = prefix + suffix
                clone_key = frame_key(clone)
                old = events.get(clone_key)
                if old is None or len(path) < len(old[1]):
                    events[clone_key] = (clone, path, kind)
        ordered = sorted(events.values(), key=lambda item: len(item[1]))
        print("reload_events", len(ordered),
              [(len(path), kind, compact(node.frame()))
               for node, path, kind in ordered], flush=True)
    elif os.environ.get("L9_MODE") == "shifted_reloads":
        events = {}
        model_summary = []
        for root_index, (root, prefix) in enumerate(roots):
            shifted = root.clone()
            for offset in range(15):
                models, states = reload_paths(shifted.frame(), max_depth=20)
                if models:
                    model_summary.append((
                        root_index, offset, states,
                        tuple((cost, kind) for cost, kind, _, _ in models),
                    ))
                for _, kind, _, suffix in models:
                    clone = shifted.clone()
                    before = arr(clone.frame()).copy()
                    for action in suffix:
                        safe_step(clone, action)
                    if frame_delta(before, clone.frame())["count"] <= 500:
                        continue
                    path = prefix + (4,) * offset + suffix
                    clone_key = frame_key(clone)
                    old = events.get(clone_key)
                    if old is None or len(path) < len(old[1]):
                        events[clone_key] = (clone, path, kind)
                safe_step(shifted, 4)
        ordered = sorted(events.values(), key=lambda item: len(item[1]))
        print("shifted_reload_models", model_summary, flush=True)
        print("shifted_reload_events", len(ordered),
              [(len(path), kind, compact(node.frame()))
               for node, path, kind in ordered], flush=True)
    elif os.environ.get("L9_MODE") == "replay_search":
        solution, explored, cost, result = extend_replay(
            roots,
            max_total_cost=int(os.environ.get("L9_TOTAL_COST", "120")),
            max_states=int(os.environ.get("L9_TOTAL_STATES", "6000")),
        )
        print("replay_search", explored, cost, solution, flush=True)
        if result is not None:
            print("replay_result", int(result.levels_completed),
                  compact(result.frame()), flush=True)
    else:
        solution, explored, cost, result = extend(
            roots,
            max_total_cost=int(os.environ.get("L9_TOTAL_COST", "56")),
            max_states=int(os.environ.get("L9_TOTAL_STATES", "12000")),
        )
        print("alternative_search", explored, cost, solution, flush=True)
        if result is not None:
            print("alternative_result", int(result.levels_completed), compact(result.frame()),
                  flush=True)


arena.run_program("lf52", probe)
