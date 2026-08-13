"""Reward-verified deletion minimization for one selected campaign level."""

import json
import math
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


TRANSITIONS = (0, 8, 42, 87, 149, 238, 331, 476, 544)
TARGET_LEVEL = int(os.environ.get("DDMIN_LEVEL", "9"))
MAX_TESTS = int(os.environ.get("DDMIN_TESTS", "400"))


def winning_prefix(entry, actions):
    clone = entry.clone()
    base_level = clone.levels_completed
    for index, action in enumerate(actions, 1):
        clone.step(action)
        if clone.levels_completed > base_level:
            return list(actions[:index])
    return None


def minimize(entry, actions):
    current = winning_prefix(entry, actions)
    tests = 1
    granularity = 2
    while current and len(current) >= 2 and tests < MAX_TESTS:
        chunk = math.ceil(len(current) / granularity)
        reduced = False
        for start in range(0, len(current), chunk):
            if tests >= MAX_TESTS:
                break
            result = winning_prefix(entry, current[:start] + current[start + chunk:])
            tests += 1
            if result is not None:
                before = len(current)
                current = result
                print("IMPROVE", before, len(current), (start, start + chunk), flush=True)
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
    with open("level9_candidate_102.json") as candidate_file:
        level9 = json.load(candidate_file)
    if TARGET_LEVEL <= 8:
        start = TRANSITIONS[TARGET_LEVEL - 1]
        end = TRANSITIONS[TARGET_LEVEL]
        for action in prefix[:start]:
            env.step(action)
        original = prefix[start:end]
    else:
        for action in prefix:
            env.step(action)
        original = level9
    entry = env.clone()
    result, tests = minimize(entry, original)
    filename = f"level{TARGET_LEVEL}_ddmin_{len(result)}.json"
    with open(filename, "w") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    verified = winning_prefix(entry, result)
    print("RESULT", {
        "level": TARGET_LEVEL,
        "original": len(original),
        "optimized": len(result),
        "saved": len(original) - len(result),
        "tests": tests,
        "verified": verified is not None and len(verified) == len(result),
        "file": filename,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
