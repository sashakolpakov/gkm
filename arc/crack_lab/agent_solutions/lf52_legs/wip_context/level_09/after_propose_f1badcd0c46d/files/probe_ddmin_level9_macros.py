"""Reward-check deletion minimization over atomic level-9 puzzle macros."""

from collections import deque
import json
import math
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


MAX_TESTS = int(os.environ.get("MACRO_DDMIN_TESTS", "160"))
TARGET_LEVEL = int(os.environ.get("MACRO_LEVEL", "9"))
ENTRY_INDEX = int(os.environ.get("MACRO_ENTRY", "544"))
EXIT_INDEX = int(os.environ.get("MACRO_EXIT", "0"))
SEED_REMOVE = os.environ.get("MACRO_SEED_REMOVE")


def macro_units(actions):
    units = []
    index = 0
    while index < len(actions):
        action = actions[index]
        if isinstance(action, list):
            if index + 1 >= len(actions) or not isinstance(actions[index + 1], list):
                raise ValueError("unpaired coordinate click")
            units.append((action, actions[index + 1]))
            index += 2
        else:
            units.append((action,))
            index += 1
    return units


def flatten(units):
    return [action for unit in units for action in unit]


def play_action(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def winning_prefix(entry, units):
    clone = entry.clone()
    base_level = clone.levels_completed
    for unit_index, unit in enumerate(units, 1):
        for action in unit:
            play_action(clone, action)
        if clone.levels_completed > base_level:
            return list(units[:unit_index])
    return None


def minimize(entry, original):
    current = winning_prefix(entry, original)
    tests = 1
    granularity = 2
    while current and len(current) >= 2 and tests < MAX_TESTS:
        chunk = math.ceil(len(current) / granularity)
        reduced = False
        for start in range(0, len(current), chunk):
            if tests >= MAX_TESTS:
                break
            trial = current[:start] + current[start + chunk:]
            result = winning_prefix(entry, trial)
            tests += 1
            if result is not None:
                before_actions = len(flatten(current))
                current = result
                print("IMPROVE", {
                    "macros": len(current),
                    "actions": (before_actions, len(flatten(current))),
                    "removed": (start, start + chunk),
                }, flush=True)
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return current, tests


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    if TARGET_LEVEL == 9:
        with open("level9_candidate_102.json") as candidate_file:
            candidate = json.load(candidate_file)
    else:
        candidate = prefix[ENTRY_INDEX:EXIT_INDEX]
    for action in prefix[:ENTRY_INDEX]:
        play_action(env, action)
    entry = env.clone()
    original = macro_units(candidate)
    if SEED_REMOVE is not None:
        remove_index = int(SEED_REMOVE)
        original = original[:remove_index] + original[remove_index + 1:]
    result, tests = minimize(entry, original)
    actions = flatten(result)
    filename = f"level{TARGET_LEVEL}_macro_ddmin_candidate.json"
    with open(filename, "w") as candidate_file:
        json.dump(actions, candidate_file, indent=2)
        candidate_file.write("\n")
    print("RESULT", {
        "original": (len(original), len(candidate)),
        "level": TARGET_LEVEL,
        "optimized": (len(result), len(actions)),
        "saved": len(candidate) - len(actions),
        "tests": tests,
        "verified": winning_prefix(entry, result) is not None,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
