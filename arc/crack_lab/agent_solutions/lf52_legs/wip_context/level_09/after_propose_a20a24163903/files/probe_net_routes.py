"""Replace key detours with shortest net-displacement interleavings."""

import json
import os

import gkm_try


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", "300"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "-1"))
EXTRA_PAIR = os.environ.get("EXTRA_PAIR") == "1"
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


def interleavings(first_action, first_count, second_action, second_count):
    if first_count == 0:
        yield (second_action,) * second_count
        return
    if second_count == 0:
        yield (first_action,) * first_count
        return
    for suffix in interleavings(first_action, first_count - 1, second_action, second_count):
        yield (first_action,) + suffix
    for suffix in interleavings(first_action, first_count, second_action, second_count - 1):
        yield (second_action,) + suffix


def multiset_routes(counts, prefix=()):
    if not any(counts.values()):
        yield prefix
        return
    for action in (1, 2, 3, 4):
        if counts.get(action, 0) <= 0:
            continue
        child = dict(counts)
        child[action] -= 1
        yield from multiset_routes(child, prefix + (action,))


def minimize(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]:
        env.step(action)
    entry = env.clone()
    segment = path[start:end]

    groups = []
    index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not is_coordinate(segment[index]):
            keys.append(segment[index])
            index += 1
        clicks = []
        while index < len(segment) and is_coordinate(segment[index]) and len(clicks) < 2:
            clicks.append(segment[index])
            index += 1
        groups.append([keys, clicks])

    trials = 0

    def succeeds(candidate):
        nonlocal trials
        trials += 1
        node = entry.clone()
        for keys, clicks in candidate:
            for action in keys + clicks:
                node.step(action)
            if node.levels_completed >= TARGET_LEVEL:
                return True
        return False

    changed = True
    while changed and trials < MAX_TRIALS:
        changed = False
        for group_index, (keys, _) in enumerate(groups):
            if TARGET_GROUP >= 0 and group_index != TARGET_GROUP:
                continue
            vertical = keys.count(2) - keys.count(1)
            horizontal = keys.count(4) - keys.count(3)
            vertical_action = 2 if vertical >= 0 else 1
            horizontal_action = 4 if horizontal >= 0 else 3
            routes = interleavings(
                vertical_action, abs(vertical), horizontal_action, abs(horizontal)
            )
            if EXTRA_PAIR:
                base_counts = {
                    vertical_action: abs(vertical),
                    horizontal_action: abs(horizontal),
                }
                route_sets = []
                for first, second in ((1, 2), (3, 4)):
                    counts = dict(base_counts)
                    counts[first] = counts.get(first, 0) + 1
                    counts[second] = counts.get(second, 0) + 1
                    route_sets.append(multiset_routes(counts))
                routes = (route for route_set in route_sets for route in route_set)
            for route in routes:
                if len(route) >= len(keys) or trials >= MAX_TRIALS:
                    continue
                candidate = [[list(k), list(c)] for k, c in groups]
                candidate[group_index][0] = list(route)
                if succeeds(candidate):
                    groups = candidate
                    changed = True
                    break
            if changed or trials >= MAX_TRIALS:
                break

    actions = [action for keys, clicks in groups for action in keys + clicks]
    print(
        "NET_RESULT", TARGET_LEVEL, len(segment), len(actions),
        trials, succeeds(groups),
    )
    print("NET_PATH", actions)


gkm_try.A.run_program("lf52", minimize)
