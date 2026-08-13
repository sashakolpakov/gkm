"""Delete redundant atomic tokens from one admitted per-level action path."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def atomic_tokens(actions):
    if os.environ.get("INDIVIDUAL_CLICKS") == "1":
        return [(action,) for action in actions]
    tokens = []
    index = 0
    while index < len(actions):
        action = actions[index]
        if isinstance(action, list) and len(action) == 3 and action[0] == 6:
            if index + 1 >= len(actions):
                raise ValueError("unpaired coordinate action")
            tokens.append((action, actions[index + 1]))
            index += 2
        else:
            tokens.append((action,))
            index += 1
    return tokens


def flatten(tokens):
    return [action for token in tokens for action in token]


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    if not 1 <= level <= 8:
        raise ValueError("TARGET_LEVEL must be in 1..8")
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in admitted[:start]:
        env.step(action)
    root = env.clone()
    base_level = int(root.levels_completed)
    tokens = atomic_tokens(admitted[start:end])
    trials = 0

    def succeeds(candidate):
        nonlocal trials
        trials += 1
        node = root.clone()
        for token in candidate:
            for action in token:
                if node.terminal():
                    break
                node.step(action)
            if node.levels_completed > base_level:
                return True
        return node.levels_completed > base_level

    assert succeeds(tokens)
    granularity = 2
    while len(tokens) >= 2:
        chunk = (len(tokens) + granularity - 1) // granularity
        reduced = False
        for first in range(0, len(tokens), chunk):
            candidate = tokens[:first] + tokens[first + chunk:]
            if candidate and succeeds(candidate):
                tokens = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                print("REDUCED", {"tokens": len(tokens),
                                  "actions": len(flatten(tokens)),
                                  "trials": trials})
                break
        if not reduced:
            if granularity >= len(tokens):
                break
            granularity = min(len(tokens), granularity * 2)

    index = 0
    while index < len(tokens):
        candidate = tokens[:index] + tokens[index + 1:]
        if candidate and succeeds(candidate):
            tokens = candidate
            print("GREEDY", {"tokens": len(tokens),
                             "actions": len(flatten(tokens)),
                             "trials": trials})
        else:
            index += 1

    candidate = flatten(tokens)
    print("CANDIDATE", {"level": level, "old_actions": end - start,
                        "actions": len(candidate), "trials": trials,
                        "path": candidate})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
