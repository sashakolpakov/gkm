"""Target reward checks at actions with no observable physical frame effect."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


TARGET_LEVEL = int(os.environ.get("INERT_LEVEL", "7"))
FILES = {
    4: "level4_greedy_macro_candidate.json",
    5: "level5_greedy_macro_candidate.json",
    6: "level6_greedy_macro_candidate.json",
    7: "level7_greedy_macro_candidate.json",
    8: "level8_greedy_macro_candidate.json",
    9: "level9_macro_ddmin_candidate.json",
}
ENTRIES = {4: 87, 5: 149, 6: 238, 7: 331, 8: 476, 9: 544}


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def physical_frame(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame


def winning_prefix(entry, actions):
    node = entry.clone()
    base_level = node.levels_completed
    for index, action in enumerate(actions, 1):
        play(node, action)
        if node.levels_completed > base_level:
            return list(actions[:index])
    return None


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:ENTRIES[TARGET_LEVEL]]:
        play(env, action)
    entry = env.clone()
    with open(FILES[TARGET_LEVEL]) as action_file:
        actions = json.load(action_file)
    node = entry.clone()
    inert = []
    for index, action in enumerate(actions):
        before = physical_frame(node)
        play(node, action)
        if np.array_equal(before, physical_frame(node)):
            inert.append(index)
    print("INERT", {
        "level": TARGET_LEVEL,
        "actions": len(actions),
        "indices": inert,
        "values": [actions[index] for index in inert],
    }, flush=True)

    current = list(actions)
    removed = []
    for original_index in inert:
        current_index = original_index - sum(
            earlier < original_index for earlier in removed
        )
        trial = current[:current_index] + current[current_index + 1:]
        result = winning_prefix(entry, trial)
        if result is not None:
            current = result
            removed.append(original_index)
            print("IMPROVE", {
                "removed_original_index": original_index,
                "length": len(current),
            }, flush=True)
    print("RESULT", {
        "level": TARGET_LEVEL,
        "original": len(actions),
        "optimized": len(current),
        "removed": removed,
        "verified": winning_prefix(entry, current) is not None,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
