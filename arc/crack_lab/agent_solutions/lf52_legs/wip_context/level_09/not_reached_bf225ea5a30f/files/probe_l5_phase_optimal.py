"""Learn one carrier maze, then solve level 5's final peg phase symbolically."""

from collections import deque
from heapq import heappop, heappush
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import safe_step


PHASE_ENTRY = 183
MAX_CARRIER_STATES = 160
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def play(env, action):
    safe_step(env, tuple(action) if isinstance(action, list) else action)


def moving_support(env):
    state = _bridge_carrier_state(env.frame())
    supports = state[3]
    candidates = {
        (border[0] + 1, border[1] + 1)
        for border in state[4]
    } & set(supports)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one visible bordered support, got {sorted(candidates)}"
        )
    return next(iter(candidates))


def carrier_graph(root):
    start = moving_support(root)
    nodes = {start: root.clone()}
    edges = {}
    queue = deque([start])
    while queue and len(nodes) <= MAX_CARRIER_STATES:
        point = queue.popleft()
        node = nodes[point]
        for action in (1, 2, 3, 4):
            child = node.clone()
            play(child, action)
            destination = moving_support(child)
            edges[(point, action)] = destination
            if destination not in nodes:
                nodes[destination] = child
                queue.append(destination)
    return start, nodes, edges, len(queue)


def shortest_carrier_path(start, target, edges):
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        point, path = queue.popleft()
        if point == target:
            return path
        for action in (1, 2, 3, 4):
            child = edges.get((point, action))
            if child is not None and child not in seen:
                seen.add(child)
                queue.append((child, path + (action,)))
    return None


def solve(start_support, graph, slots, fixed_supports, start_pegs):
    start = (start_support, frozenset(start_pegs))
    serial = 0
    queue = [(0, serial, start)]
    best = {start: 0}
    parent = {}
    goal = None
    while queue:
        cost, _, state = heappop(queue)
        if cost != best[state]:
            continue
        support, pegs = state
        if len(pegs) == 1:
            goal = state
            break

        for action in (1, 2, 3, 4):
            child_support = graph.get((support, action))
            if child_support is None or child_support == support:
                continue
            child = (child_support, pegs)
            child_cost = cost + 1
            if child_cost < best.get(child, 10 ** 9):
                best[child] = child_cost
                parent[child] = (state, action)
                serial += 1
                heappush(queue, (child_cost, serial, child))

        occupied_supports = fixed_supports | {support}
        for source in sorted(pegs):
            for dr, dc in DIRECTIONS:
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in pegs | occupied_supports
                    or destination not in slots
                    or destination in pegs | occupied_supports
                ):
                    continue
                child_pegs = set(pegs)
                child_pegs.remove(source)
                child_pegs.add(destination)
                child_pegs.discard(midpoint)
                child = (support, frozenset(child_pegs))
                child_cost = cost + 2
                if child_cost < best.get(child, 10 ** 9):
                    best[child] = child_cost
                    parent[child] = (state, (source, destination))
                    serial += 1
                    heappush(queue, (child_cost, serial, child))

    if goal is None:
        return None, len(best)
    path = []
    state = goal
    while state != start:
        state, action = parent[state]
        path.append(action)
    path.reverse()
    return (best[goal], path, goal), len(best)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:PHASE_ENTRY]:
        play(env, action)
    root = env.clone()
    state = _bridge_carrier_state(root.frame())
    slots = set(state[0]) | set(state[1])
    start_support = moving_support(root)
    fixed_supports = set(state[3]) - {start_support}
    start, nodes, edges, remaining = carrier_graph(root)
    checks = {
        target: shortest_carrier_path(start, target, edges)
        for target in ((24, 15), (24, 39), (12, 45), (12, 27), (36, 27))
    }
    result, symbolic_states = solve(
        start_support, edges, slots, fixed_supports, state[1]
    )
    print("CARRIER_GRAPH", {
        "start": start,
        "states": len(nodes),
        "remaining": remaining,
        "checks": checks,
    }, flush=True)
    print("SYMBOLIC", {
        "slots": sorted(slots),
        "fixed": sorted(fixed_supports),
        "pegs": sorted(state[1]),
        "states": symbolic_states,
        "result": result,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
