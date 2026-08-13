"""Goal-directed whole-level search for a shorter level-4 route."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves
from perception import arr, connected_components, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def peg_positions(frame):
    return frozenset(
        blob.top_left
        for blob in connected_components(frame, colors=(14,))
        if blob.size == (4, 4)
    )


def macros(frame):
    actions = [(action,) for action in (1, 2, 3, 4)]
    actions.extend(
        ((6, source[1] + 1, source[0] + 1),
         (6, destination[1] + 1, destination[0] + 1))
        for _, source, destination in _bridge_carrier_moves(frame)
    )
    return tuple(actions)


def closeness(pegs):
    ordered = sorted(pegs)
    distances = [
        abs(first[0] - second[0]) + abs(first[1] - second[1])
        for index, first in enumerate(ordered)
        for second in ordered[index + 1:]
    ]
    return min(distances) if distances else 0


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "4"))
    max_states = int(os.environ.get("OPT_STATES", "1500"))
    cost_bound = int(os.environ.get("OPT_COST", "50"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    root = None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            root = env.clone()
            break
        prior = current

    root_pegs = peg_positions(root.frame())
    serial = 0
    queue = [((0, 0, len(root_pegs), closeness(root_pegs)),
              serial, 0, 0, root.clone(), ())]
    best = {key(root): 0}
    expanded = 0
    while queue and len(best) <= max_states:
        _, _, cost, captures, node, path = heappop(queue)
        if cost != best.get(key(node)) or cost >= cost_bound:
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("progress", expanded, len(best), cost, captures, flush=True)
        before_pegs = peg_positions(node.frame())
        for macro in macros(node.frame()):
            child_cost = cost + len(macro)
            if child_cost > cost_bound:
                continue
            child = node.clone()
            before_key = key(child)
            for action in macro:
                safe_step(child, action)
                if int(child.levels_completed) >= desired:
                    solution = path + macro
                    print("solution", len(solution), len(best), expanded,
                          solution, flush=True)
                    return
            child_key = key(child)
            if child_key == before_key:
                continue
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            after_pegs = peg_positions(child.frame())
            child_captures = captures + max(0, len(before_pegs) - len(after_pegs))
            priority = (
                child_cost - 12 * child_captures,
                -child_captures,
                len(after_pegs),
                closeness(after_pegs),
            )
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (priority, serial, child_cost, child_captures,
                             child, path + macro))
    print("none", len(best), expanded, flush=True)


arena.run_program("lf52", probe)
