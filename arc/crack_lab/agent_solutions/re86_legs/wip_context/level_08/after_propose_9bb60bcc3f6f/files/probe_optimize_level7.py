import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr


PREFIX = 306


def step_many(node, path):
    for action in path:
        node.step(action)


def erase_loops(root, route):
    node = root.clone()
    states = [arr(node.frame()).tobytes()]
    for action in route:
        node.step(action)
        states.append(arr(node.frame()).tobytes())
    latest = {}
    best = None
    for index, state in enumerate(states):
        if state in latest:
            gap = index - latest[state]
            if best is None or gap > best[0]:
                best = (gap, latest[state], index)
        latest[state] = index
    if best is None:
        return route, None
    gap, start, end = best
    return route[:start] + route[end:], (start, end, gap)


def runs(route):
    result = []
    for action in route:
        if result and result[-1][0] == action:
            result[-1] = (action, result[-1][1] + 1)
        else:
            result.append((action, 1))
    return result


def completes(root, route):
    node = root.clone()
    step_many(node, route)
    return node.levels_completed > root.levels_completed


def delete_chunks(root, route):
    for width in (32, 16, 8, 4, 2, 1):
        index = 0
        while index < len(route):
            candidate = route[:index] + route[index + width :]
            if completes(root, candidate):
                route = candidate
                print("delete", width, index, "length", len(route), flush=True)
                index = max(0, index - width)
            else:
                index += width
    return route


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    step_many(env, full[:PREFIX])
    root = env.clone()
    route = full[PREFIX:]
    while True:
        route, removed = erase_loops(root, route)
        if removed is None:
            break
        print("loop", removed, "length", len(route))
    route = delete_chunks(root, route)
    check = root.clone()
    step_many(check, route)
    print(
        "result",
        len(route),
        "level",
        check.levels_completed,
        "runs",
        runs(route),
    )


if __name__ == "__main__":
    A.run_program("re86", probe)
