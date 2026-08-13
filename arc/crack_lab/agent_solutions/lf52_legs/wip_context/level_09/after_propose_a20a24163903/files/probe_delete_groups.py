"""Greedily delete whole validated macro groups with full reward replay."""

import json
import os

import gkm_try


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", "60"))
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
        group = []
        while index < len(segment) and not is_coordinate(segment[index]):
            group.append(segment[index])
            index += 1
        while index < len(segment) and is_coordinate(segment[index]) and sum(is_coordinate(a) for a in group) < 2:
            group.append(segment[index])
            index += 1
        groups.append(group)

    trials = 0

    def succeeds(candidate_groups):
        nonlocal trials
        trials += 1
        node = entry.clone()
        for group in candidate_groups:
            for action in group:
                node.step(action)
            if node.levels_completed >= TARGET_LEVEL:
                return True
        return False

    changed = True
    while changed and trials < MAX_TRIALS:
        changed = False
        for group_index in range(len(groups)):
            if trials >= MAX_TRIALS:
                break
            candidate = groups[:group_index] + groups[group_index + 1:]
            if succeeds(candidate):
                groups = candidate
                changed = True
                break

    actions = [action for group in groups for action in group]
    print(
        "GROUP_RESULT", TARGET_LEVEL, len(segment), len(actions),
        len(groups), trials, succeeds(groups),
    )
    print("GROUP_PATH", actions)


gkm_try.A.run_program("lf52", minimize)
