"""Learn level 7's final carrier maze and solve its piece relay exactly."""

from collections import deque
from heapq import heappop, heappush
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


LEVEL_ENTRY = 331
PHASE_ACTIONS = 109
MAX_GEOMETRY_STATES = 300
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def play(env, action):
    safe_step(env, tuple(action) if isinstance(action, list) else action)


def carriers(env):
    # The bridge parser also normalizes 2x2 clipped carrier interiors at a
    # screen edge; the movable-board parser intentionally keeps only full
    # 4x4 interiors.
    return tuple(sorted(_bridge_carrier_state(env.frame())[2]))


def carrier_graph(root):
    start = carriers(root)
    nodes = {start: root.clone()}
    edges = {}
    queue = deque([start])
    while queue and len(nodes) <= MAX_GEOMETRY_STATES:
        state = queue.popleft()
        node = nodes[state]
        for action in (1, 2, 3, 4):
            child = node.clone()
            play(child, action)
            destination = carriers(child)
            if len(destination) != len(start):
                raise RuntimeError(
                    f"empty carrier count changed: {state} -> {destination}"
                )
            edges[(state, action)] = destination
            if destination not in nodes:
                nodes[destination] = child
                queue.append(destination)
    return start, nodes, edges, len(queue)


def carrier_mapping(before, after):
    remaining = set(after)
    mapping = {}
    for source in before:
        choices = sorted(
            point for point in remaining
            if abs(point[0] - source[0]) + abs(point[1] - source[1]) <= 6
        )
        if len(choices) != 1:
            raise RuntimeError(f"ambiguous carrier edge: {before} -> {after}")
        mapping[source] = choices[0]
        remaining.remove(choices[0])
    return mapping


def solve(start_carriers, graph, slots, fixed, start_bridge, start_pegs):
    start = (start_carriers, start_bridge, frozenset(start_pegs))
    serial = 0
    queue = [(0, serial, start)]
    best = {start: 0}
    parent = {}
    goal = None
    while queue:
        cost, _, state = heappop(queue)
        if cost != best[state]:
            continue
        geometry, bridge, pegs = state
        if len(pegs) == 1:
            goal = state
            break

        occupied = pegs | {bridge}
        for action in (1, 2, 3, 4):
            child_geometry = graph.get((geometry, action))
            if child_geometry is None or child_geometry == geometry:
                continue
            mapping = carrier_mapping(geometry, child_geometry)
            child_bridge = mapping.get(bridge, bridge)
            child_pegs = frozenset(mapping.get(peg, peg) for peg in pegs)
            if len(child_pegs | {child_bridge}) != len(pegs) + 1:
                continue
            child = (child_geometry, child_bridge, child_pegs)
            child_cost = cost + 1
            if child_cost < best.get(child, 10 ** 9):
                best[child] = child_cost
                parent[child] = (state, action)
                serial += 1
                heappush(queue, (child_cost, serial, child))

        destinations = slots | set(geometry)
        for kind, pieces in (("bridge", (bridge,)), ("peg", sorted(pegs))):
            for source in pieces:
                for dr, dc in DIRECTIONS:
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                    if (
                        midpoint not in occupied | fixed
                        or destination not in destinations
                        or destination in occupied | fixed
                    ):
                        continue
                    child_bridge = bridge
                    child_pegs = set(pegs)
                    if kind == "bridge":
                        child_bridge = destination
                    else:
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    child = (geometry, child_bridge, frozenset(child_pegs))
                    child_cost = cost + 2
                    if child_cost < best.get(child, 10 ** 9):
                        best[child] = child_cost
                        parent[child] = (
                            state, (kind, source, destination)
                        )
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
    with open("level7_greedy_macro_candidate.json") as stream:
        candidate = json.load(stream)
    for action in campaign[:LEVEL_ENTRY]:
        play(env, action)
    for action in candidate[:PHASE_ACTIONS]:
        play(env, action)
    root = env.clone()
    slots, start_carriers, bridges, pegs = _movable_bridge_board(root.frame())
    bridge_state = _bridge_carrier_state(root.frame())
    fixed = set(bridge_state[3])
    slots = set(slots) | set(pegs) | set(bridges)
    if len(bridges) != 1:
        raise RuntimeError(f"expected one movable bridge, got {sorted(bridges)}")
    start, nodes, edges, remaining = carrier_graph(root)
    result, symbolic_states = solve(
        start, edges, slots, fixed, next(iter(bridges)), pegs
    )
    print("GEOMETRY", {
        "start": start,
        "states": len(nodes),
        "remaining": remaining,
        "edges": len(edges),
    }, flush=True)
    print("BOARD", {
        "slots": sorted(slots),
        "fixed": sorted(fixed),
        "bridge": sorted(bridges),
        "pegs": sorted(pegs),
    }, flush=True)
    print("SYMBOLIC", {
        "states": symbolic_states,
        "result": result,
        "known_suffix_cost": len(candidate) - PHASE_ACTIONS,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
