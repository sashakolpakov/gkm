"""Test individual key deletions with whole-level reward validation."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    return tuple(groups)


def flatten(groups):
    return tuple(action for keys, clicks in groups for action in keys + clicks)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    selected = tuple(int(value) for value in
                     os.environ.get("OPT_GROUPS", "20,21,22").split(","))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current
    groups = list(split(campaign[start:end]))
    successes = []
    tests = 0
    greedy = os.environ.get("OPT_GREEDY") == "1"
    delete_clicks = os.environ.get("OPT_DELETE_CLICKS") == "1"
    delete_half_clicks = os.environ.get("OPT_DELETE_HALF_CLICKS") == "1"
    delete_pairs = os.environ.get("OPT_DELETE_PAIRS") == "1"
    alternate_destination = os.environ.get("OPT_ALT_DEST") == "1"

    def wins(candidate_groups):
        node = entry.clone()
        for action in flatten(candidate_groups):
            safe_step(node, action)
            if int(node.levels_completed) >= desired:
                return True
        return False

    for group_index in selected:
        if alternate_destination:
            keys, clicks = groups[group_index]
            source, destination = clicks
            alternatives = []
            for dx, dy in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                candidate = (6, destination[1] + dx,
                             destination[2] + dy)
                if (
                    candidate != destination
                    and 0 <= candidate[1] <= 63
                    and 0 <= candidate[2] <= 63
                ):
                    alternatives.append(candidate)
            for candidate in alternatives:
                candidate_groups = list(groups)
                candidate_groups[group_index] = (
                    keys, (source, candidate)
                )
                won = wins(candidate_groups)
                tests += 1
                if won:
                    successes.append((group_index, destination, candidate,
                                      flatten(candidate_groups)))
                    print("success", tests, group_index, destination,
                          candidate, len(flatten(candidate_groups)),
                          flush=True)
            continue
        if delete_half_clicks:
            click_index = 0
            while click_index < len(groups[group_index][1]):
                keys, clicks = groups[group_index]
                removed = clicks[click_index]
                candidate_groups = list(groups)
                candidate_groups[group_index] = (
                    keys, clicks[:click_index] + clicks[click_index + 1:]
                )
                won = wins(candidate_groups)
                tests += 1
                if won:
                    successes.append((group_index, click_index, removed,
                                      flatten(candidate_groups)))
                    print("success", tests, group_index, click_index, removed,
                          len(flatten(candidate_groups)), flush=True)
                    if greedy:
                        groups = candidate_groups
                    else:
                        click_index += 1
                else:
                    click_index += 1
            continue
        if delete_clicks:
            keys, clicks = groups[group_index]
            candidate_groups = list(groups)
            candidate_groups[group_index] = (keys, ())
            won = wins(candidate_groups)
            if won:
                successes.append((group_index, -1, 6,
                                  flatten(candidate_groups)))
                if greedy:
                    groups = candidate_groups
            tests += 1
            print("test", tests, group_index, "clicks", won,
                  len(flatten(groups)), flush=True)
            continue
        if delete_pairs:
            while True:
                keys, clicks = groups[group_index]
                found = False
                for first in range(len(keys)):
                    for second in range(first + 1, len(keys)):
                        candidate_groups = list(groups)
                        candidate_groups[group_index] = (
                            keys[:first] + keys[first + 1:second]
                            + keys[second + 1:], clicks
                        )
                        won = wins(candidate_groups)
                        tests += 1
                        if won:
                            successes.append((group_index, (first, second),
                                              (keys[first], keys[second]),
                                              flatten(candidate_groups)))
                            print("success", tests, group_index,
                                  (first, second), (keys[first], keys[second]),
                                  len(flatten(candidate_groups)), flush=True)
                            if greedy:
                                groups = candidate_groups
                                found = True
                                break
                    if found:
                        break
                if not greedy or not found:
                    break
            print("group", group_index, "tests", tests,
                  "keys", len(groups[group_index][0]), flush=True)
            continue
        key_index = 0
        while key_index < len(groups[group_index][0]):
            keys, clicks = groups[group_index]
            removed = keys[key_index]
            candidate_groups = list(groups)
            candidate_groups[group_index] = (
                keys[:key_index] + keys[key_index + 1:], clicks
            )
            won = wins(candidate_groups)
            if won:
                successes.append((group_index, key_index, removed,
                                  flatten(candidate_groups)))
                if greedy:
                    groups = candidate_groups
                else:
                    key_index += 1
            else:
                key_index += 1
            tests += 1
            print("test", tests, group_index, key_index, won,
                  len(flatten(groups)), flush=True)
    print("successes", tuple((g, i, action, len(path))
                             for g, i, action, path in successes), flush=True)
    if successes:
        print("best", successes[0][3], flush=True)


arena.run_program("lf52", probe)
