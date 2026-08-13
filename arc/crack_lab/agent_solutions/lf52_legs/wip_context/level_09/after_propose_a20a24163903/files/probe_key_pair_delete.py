"""Remove coordinated inverse key pairs with full-level reward validation."""

import json
import os

import gkm_try


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", "240"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}
INVERSE = {1: 2, 2: 1, 3: 4, 4: 3}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


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
            for first in range(len(keys)):
                for second in range(first + 1, len(keys)):
                    if keys[second] != INVERSE.get(keys[first]) or trials >= MAX_TRIALS:
                        continue
                    candidate = [[list(k), list(c)] for k, c in groups]
                    candidate[group_index][0] = [
                        action for index, action in enumerate(keys)
                        if index not in (first, second)
                    ]
                    if succeeds(candidate):
                        groups = candidate
                        changed = True
                        break
                if changed or trials >= MAX_TRIALS:
                    break
            if changed or trials >= MAX_TRIALS:
                break

    actions = [action for keys, clicks in groups for action in keys + clicks]
    print(
        "PAIR_RESULT", TARGET_LEVEL, len(segment), len(actions),
        trials, succeeds(groups),
    )
    print("PAIR_PATH", actions)


gkm_try.A.run_program("lf52", minimize)
