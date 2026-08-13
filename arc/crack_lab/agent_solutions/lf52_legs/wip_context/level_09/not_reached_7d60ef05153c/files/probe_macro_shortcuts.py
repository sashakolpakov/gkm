"""Search bounded whole-macro shortcuts between admitted trajectory states."""

import heapq
import json
import os

import gkm_try
from legs import _bridge_carrier_moves
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
START_GROUP = int(os.environ.get("START_GROUP", "-1"))
MAX_STATES = int(os.environ.get("MAX_STATES", "2500"))
MAX_COST = int(os.environ.get("MAX_COST", "24"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_click(action):
    return isinstance(action, (list, tuple)) and len(action) == 3


def key(node):
    return arr(node.frame())[1:, :].tobytes()


def piece_macros(node):
    if TARGET_LEVEL <= 5:
        for _, source, destination in _bridge_carrier_moves(node.frame()):
            yield (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
        return
    sources = []
    for blob in connected_components(node.frame(), colors=(8, 9, 14)):
        if blob.size == (4, 4) and blob.area >= 12:
            sources.append(blob.top_left)
    for row, col in sorted(set(sources)):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = row + dr, col + dc
            if not (0 <= destination[0] <= 60 and 0 <= destination[1] <= 60):
                continue
            yield (
                (6, col + 1, row + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )


def search(root, targets, base_level):
    serial = 0
    root_key = key(root)
    distance = {root_key: 0}
    queue = [(0, serial, root.clone(), ())]
    best = None
    while queue and len(distance) <= MAX_STATES:
        cost, _, node, path = heapq.heappop(queue)
        node_key = key(node)
        if cost != distance.get(node_key) or cost >= MAX_COST:
            continue
        target = targets.get(node_key)
        if target is not None:
            target_group, original_cost = target
            saving = original_cost - cost
            if saving > 0 and (best is None or saving > best[0]):
                best = saving, target_group, cost, path
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += tuple(piece_macros(node))
        for macro in macros:
            child = node.clone()
            for action in macro:
                if isinstance(action, tuple):
                    child.step(*action)
                else:
                    child.step(action)
            child_key = key(child)
            child_cost = cost + len(macro)
            if child.levels_completed > base_level:
                return (10 ** 6, -1, child_cost, path + macro), len(distance)
            if child_key == node_key or child_cost >= distance.get(child_key, 10 ** 9):
                continue
            distance[child_key] = child_cost
            serial += 1
            heapq.heappush(queue, (child_cost, serial, child, path + macro))
    return best, len(distance)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start]:
        env.step(action)
    entry = env.clone()
    segment = full_path[start:end]
    groups = []
    index = 0
    while index < len(segment):
        group = []
        while index < len(segment) and not is_click(segment[index]):
            group.append(segment[index]); index += 1
        while index < len(segment) and is_click(segment[index]) and sum(is_click(a) for a in group) < 2:
            group.append(tuple(segment[index])); index += 1
        groups.append(tuple(group))

    nodes = [entry.clone()]
    costs = [0]
    node = entry.clone()
    for group in groups:
        for action in group:
            if isinstance(action, tuple): node.step(*action)
            else: node.step(action)
        nodes.append(node.clone())
        costs.append(costs[-1] + len(group))

    starts = range(len(groups)) if START_GROUP < 0 else (START_GROUP,)
    overall = None
    total_states = 0
    for first in starts:
        targets = {
            key(nodes[last]): (last, costs[last] - costs[first])
            for last in range(first + 1, len(nodes))
        }
        result, states = search(nodes[first], targets, TARGET_LEVEL - 1)
        total_states += states
        if result is not None and (overall is None or result[0] > overall[0]):
            overall = (result[0], first, *result[1:])
    valid = False
    candidate = None
    if overall is not None:
        _, first, last, _, shortcut = overall
        candidate = tuple(action for group in groups[:first] for action in group)
        candidate += shortcut
        if last >= 0:
            candidate += tuple(action for group in groups[last:] for action in group)
        validation = entry.clone()
        for action in candidate:
            if isinstance(action, tuple): validation.step(*action)
            else: validation.step(action)
            if validation.levels_completed >= TARGET_LEVEL: break
        valid = validation.levels_completed >= TARGET_LEVEL
    print("SHORTCUT_RESULT", TARGET_LEVEL, START_GROUP, len(segment), overall, total_states, valid)
    if valid:
        print("SHORTCUT_PATH", candidate)


gkm_try.A.run_program("lf52", probe)
