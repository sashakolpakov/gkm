"""Greedily delete key or click portions of validated macro groups."""

import json
import os
from math import ceil

import gkm_try


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", "80"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


def minimize(env):
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
        for group_index, group in enumerate(groups):
            for part_index in (0, 1):
                if not group[part_index] or trials >= MAX_TRIALS:
                    continue
                candidate = [[list(keys), list(clicks)] for keys, clicks in groups]
                candidate[group_index][part_index] = []
                if succeeds(candidate):
                    groups = candidate
                    changed = True
                    break
            if changed or trials >= MAX_TRIALS:
                break

    actions = [action for keys, clicks in groups for action in keys + clicks]

    def succeeds_actions(candidate_actions):
        nonlocal trials
        trials += 1
        node = entry.clone()
        for action in candidate_actions:
            node.step(action)
            if node.levels_completed >= TARGET_LEVEL:
                return True
        return False

    granularity = 2
    while len(actions) >= 2 and trials < MAX_TRIALS:
        chunk_size = ceil(len(actions) / granularity)
        reduced = False
        for start_index in range(0, len(actions), chunk_size):
            if trials >= MAX_TRIALS:
                break
            candidate = actions[:start_index] + actions[start_index + chunk_size:]
            if succeeds_actions(candidate):
                actions = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(actions):
            break
        granularity = min(len(actions), granularity * 2)

    changed = True
    while changed and trials < MAX_TRIALS:
        changed = False
        for action_index in range(len(actions)):
            if trials >= MAX_TRIALS:
                break
            candidate = actions[:action_index] + actions[action_index + 1:]
            if succeeds_actions(candidate):
                actions = candidate
                changed = True
                break
    print(
        "PART_RESULT", TARGET_LEVEL, len(segment), len(actions),
        trials, succeeds_actions(actions),
    )
    print("PART_PATH", actions)


gkm_try.A.run_program("lf52", minimize)
