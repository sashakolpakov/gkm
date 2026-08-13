"""Greedily test individual atomic-macro deletions in a bounded index range."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


ENTRY_INDEX = int(os.environ.get("GREEDY_ENTRY", "331"))
START_INDEX = int(os.environ.get("GREEDY_START", "0"))
END_INDEX = int(os.environ.get("GREEDY_END", "40"))
BASE_FILE = os.environ.get(
    "GREEDY_BASE", "level7_macro_ddmin_candidate.json"
)
OUTPUT_FILE = os.environ.get(
    "GREEDY_OUTPUT", "level7_greedy_macro_candidate.json"
)


def macro_units(actions):
    units = []
    index = 0
    while index < len(actions):
        action = actions[index]
        if isinstance(action, list):
            units.append((action, actions[index + 1]))
            index += 2
        else:
            units.append((action,))
            index += 1
    return units


def flatten(units):
    return [action for unit in units for action in unit]


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def winning_prefix(entry, units):
    node = entry.clone()
    base_level = node.levels_completed
    for unit_index, unit in enumerate(units, 1):
        for action in unit:
            play(node, action)
        if node.levels_completed > base_level:
            return list(units[:unit_index])
    return None


def save(units):
    with open(OUTPUT_FILE, "w") as output_file:
        json.dump(flatten(units), output_file, indent=2)
        output_file.write("\n")


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    with open(BASE_FILE) as base_file:
        current = macro_units(json.load(base_file))
    for action in campaign[:ENTRY_INDEX]:
        play(env, action)
    entry = env.clone()
    index = START_INDEX
    tests = 0
    while index < min(END_INDEX, len(current)):
        tests += 1
        result = winning_prefix(
            entry, current[:index] + current[index + 1:]
        )
        if result is not None:
            before = len(flatten(current))
            current = result
            save(current)
            print("IMPROVE", {
                "index": index,
                "actions": (before, len(flatten(current))),
                "file": OUTPUT_FILE,
            }, flush=True)
            continue
        index += 1
    save(current)
    print("RESULT", {
        "range": (START_INDEX, END_INDEX),
        "tests": tests,
        "actions": len(flatten(current)),
        "verified": winning_prefix(entry, current) is not None,
        "file": OUTPUT_FILE,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
