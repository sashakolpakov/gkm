"""Bounded best-first search using public action 7 for exact backtracking."""

import json
import os
from heapq import heappop, heappush

import gkm_try
from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
MAX_STATES = int(os.environ.get("MAX_STATES", "3000"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def play(node, action):
    if isinstance(action, tuple):
        node.step(*action)
    else:
        node.step(action)


def search(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    prefix = full_path[:LEVEL_ENDS[TARGET_LEVEL - 1]]
    for action in prefix:
        env.step(action)
    entry = env.clone()
    validation_root = env.clone()

    def key(node):
        return node.levels_completed, arr(node.frame())[1:, :].tobytes()

    def priority(node, cost):
        slots, pegs = _bridge_carrier_state(node.frame())[:2]
        return cost + 8 * len(pegs) - len(slots)

    current_path = ()

    def navigate(target_path):
        nonlocal current_path
        common = 0
        while (
            common < len(current_path)
            and common < len(target_path)
            and current_path[common] == target_path[common]
        ):
            common += 1
        for _ in current_path[common:]:
            entry.step(7)
        for action in target_path[common:]:
            play(entry, action)
        current_path = target_path

    root_key = key(entry)
    serial = 0
    queue = [(priority(entry, 0), 0, serial, (), root_key)]
    best_cost = {root_key: 0}
    best_dense = (len(_bridge_carrier_state(entry.frame())[1]), len(_bridge_carrier_state(entry.frame())[0]))
    best_path = ()
    solution = None

    while queue and len(best_cost) <= MAX_STATES:
        _, cost, _, path, expected_key = heappop(queue)
        if cost != best_cost.get(expected_key):
            continue
        navigate(path)
        if key(entry) != expected_key:
            raise RuntimeError(
                f"undo navigation failed current={current_path!r} queued={path!r}"
            )
        if entry.levels_completed >= TARGET_LEVEL:
            solution = path
            break

        before_key = expected_key
        macros = [(action,) for action in (1, 2, 3, 4)]
        macros += [
            (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            for _, source, destination in _bridge_carrier_moves(entry.frame())
        ]
        for macro in macros:
            successful_steps = 0
            for action in macro:
                before_action = (
                    entry.levels_completed,
                    arr(entry.frame())[1:, :].tobytes(),
                )
                play(entry, action)
                if key(entry) != before_action:
                    successful_steps += 1
            child_key = key(entry)
            child_path = path + macro
            child_cost = cost + len(macro)
            if entry.levels_completed >= TARGET_LEVEL:
                solution = child_path
                break
            if child_key != before_key and child_cost < best_cost.get(child_key, 10 ** 9):
                best_cost[child_key] = child_cost
                state = _bridge_carrier_state(entry.frame())
                dense = (len(state[1]), -len(state[0]))
                if dense < best_dense:
                    best_dense = dense
                    best_path = child_path
                serial += 1
                heappush(
                    queue,
                    (priority(entry, child_cost), child_cost, serial, child_path, child_key),
                )
            for _ in range(successful_steps):
                entry.step(7)
        if solution is not None:
            break

    valid = False
    if solution is not None:
        for action in solution:
            play(validation_root, action)
        valid = validation_root.levels_completed >= TARGET_LEVEL
    print(
        "INPLACE_RESULT", TARGET_LEVEL, len(best_cost), best_dense,
        None if solution is None else len(solution), valid,
    )
    print("INPLACE_PATH", solution)
    print("INPLACE_DENSE_PATH", best_path)


levels, path, error = gkm_try.A.run_program("lf52", search)
print("SEARCH_RUN", levels, len(path), error)
