"""Reward-test replacing directional key spans with one neutral turn."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def winning_prefix(entry, actions):
    node = entry.clone()
    base_level = int(node.levels_completed)
    for index, action in enumerate(actions, 1):
        play(node, action)
        if node.levels_completed > base_level:
            return list(actions[:index])
    return None


def key_spans(actions):
    runs = []
    start = None
    for index, action in enumerate(actions + [None]):
        if isinstance(action, int):
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    spans = []
    for first, last in runs:
        for length in range(last - first, 1, -1):
            for start in range(first, last - length + 1):
                spans.append((length - 1, start, start + length))
    return sorted(spans, reverse=True)


def probe(env):
    level = int(os.environ.get("TARGET_LEVEL", "7"))
    candidate_name = os.environ.get(
        "CANDIDATE", f"level{level}_greedy_macro_candidate.json"
    )
    max_tests = int(os.environ.get("MAX_TESTS", "300"))
    output_name = os.environ.get(
        "OUTPUT", f"level{level}_neutral_replacement_candidate.json"
    )
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:BOUNDARIES[level - 1]]:
        play(env, action)
    entry = env.clone()
    with open(candidate_name) as stream:
        current = json.load(stream)
    if winning_prefix(entry, current) is None:
        raise RuntimeError("candidate does not solve target level")

    tests = 0
    improvements = []
    while tests < max_tests:
        improved = False
        for saving, first, last in key_spans(current):
            if tests >= max_tests:
                break
            tests += 1
            trial = current[:first] + [7] + current[last:]
            result = winning_prefix(entry, trial)
            if result is None:
                continue
            improvements.append({
                "range": (first, last),
                "saving": len(current) - len(result),
            })
            current = result
            improved = True
            print("IMPROVE", {
                "test": tests,
                "actions": len(current),
                "change": improvements[-1],
            }, flush=True)
            break
        if not improved:
            break

    with open(output_name, "w") as stream:
        json.dump(current, stream, indent=2)
        stream.write("\n")
    print("RESULT", {
        "level": level,
        "actions": len(current),
        "tests": tests,
        "improvements": improvements,
        "verified": winning_prefix(entry, current) is not None,
        "file": output_name,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error),
})
