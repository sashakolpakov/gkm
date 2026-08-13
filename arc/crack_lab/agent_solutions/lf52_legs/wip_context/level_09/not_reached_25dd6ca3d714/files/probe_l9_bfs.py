"""Cost-bounded BFS over experimentally exposed level-9 click macros."""

import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


KEYS = (1, 2, 3, 4, 7)
POINTS = tuple((row, col) for row in range(0, 61, 6)
               for col in range(0, 61, 6))


def state_key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def playable_sources(frame):
    data = arr(frame)
    return tuple(
        (row, col)
        for row, col in POINTS
        if row + 2 < 64 and col + 2 < 64
        and int(data[row + 1, col + 1]) in (9, 14)
    )


def highlighted_destinations(frame):
    data = arr(frame)
    return tuple(
        (row, col)
        for row, col in POINTS
        if (data[row:row + 4, col:col + 4] == 2).any()
    )


def click(row, col):
    return (6, col + 1, row + 1)


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    base_level = int(env.levels_completed)
    serial = 0
    queue = [(0, serial, env.clone(), ())]
    best = {state_key(env): 0}
    expanded = 0
    solution = None
    max_cost, max_states = 56, 20000

    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if best.get(state_key(node)) != cost:
            continue
        if node.levels_completed > base_level:
            solution = path
            break
        expanded += 1

        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            child_cost = cost + 1
            key = state_key(child)
            if child_cost <= max_cost and child_cost < best.get(key, max_cost + 1):
                best[key] = child_cost
                serial += 1
                heappush(queue, (child_cost, serial, child, path + (action,)))

        if cost + 2 > max_cost:
            continue
        parent_key = state_key(node)
        for source_row, source_col in playable_sources(node.frame()):
            selected = node.clone()
            source_action = click(source_row, source_col)
            safe_step(selected, source_action)
            if state_key(selected) == parent_key:
                continue
            for dest_row, dest_col in highlighted_destinations(selected.frame()):
                destination_action = click(dest_row, dest_col)
                child = selected.clone()
                safe_step(child, destination_action)
                key = state_key(child)
                child_cost = cost + 2
                if key == parent_key or child_cost >= best.get(key, max_cost + 1):
                    continue
                best[key] = child_cost
                serial += 1
                child_path = path + (source_action, destination_action)
                if child.levels_completed > base_level:
                    solution = child_path
                    queue.clear()
                    break
                heappush(queue, (child_cost, serial, child, child_path))
            if solution is not None:
                break

    print("SEARCH", {"states": len(best), "expanded": expanded,
                     "remaining": len(queue), "solution_cost":
                     None if solution is None else len(solution)})
    print("SOLUTION", solution)


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
