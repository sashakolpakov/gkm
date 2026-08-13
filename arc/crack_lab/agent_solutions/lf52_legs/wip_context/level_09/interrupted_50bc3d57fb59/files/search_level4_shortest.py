"""Bounded macro shortest-path search from the validated level-4 entry."""

from collections import Counter
from heapq import heappop, heappush
import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr, safe_step


PREFIX_END = 87


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def heuristic(env):
    pegs = _bridge_carrier_state(env.frame())[1]
    return 2 * max(0, len(pegs) - 1)


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"][:PREFIX_END]:
        env.step(action)
    base_level = int(env.levels_completed)
    root = env.clone()
    queue = [(heuristic(root), 0, 0, root, ())]
    best = {key(root): 0}
    counts = Counter()
    serial = 0
    solution = None
    while queue and len(best) <= 20000:
        _, cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)):
            continue
        counts[cost] += 1
        if node.levels_completed > base_level:
            solution = path
            break
        if cost >= 62:
            continue
        successors = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            if key(child) != key(node):
                successors.append((child, (action,)))
        for _, source, destination in _bridge_carrier_moves(node.frame()):
            macro = (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            if key(child) != key(node):
                successors.append((child, macro))
        for child, macro in successors:
            child_cost = cost + len(macro)
            if child_cost > 62:
                continue
            child_key = key(child)
            if child_cost < best.get(child_key, 10 ** 9):
                best[child_key] = child_cost
                serial += 1
                heappush(
                    queue,
                    (child_cost + heuristic(child), child_cost, serial, child, path + macro),
                )
    print("LEVEL4_SEARCH", len(best), sum(counts.values()), tuple(sorted(counts.items())))
    print("LEVEL4_SOLUTION", solution)
    if solution is not None:
        clone = root.clone()
        for action in solution:
            safe_step(clone, action)
        print("LEVEL4_REPLAY", clone.levels_completed, len(solution))


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
