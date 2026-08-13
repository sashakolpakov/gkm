"""Verify undo/slide compression of one checkpoint carrier alignment."""

import json
import os

import gkm_try

from perception import arr, safe_step


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "9"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def groups(segment):
    result = []; index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        result.append((tuple(keys), tuple(pair)))
    return result


def runs(keys):
    return tuple(action for index, action in enumerate(keys) if index == 0 or action != keys[index - 1])


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full = json.load(checkpoint_file)["final_path"]
    start, end = LEVEL_ENDS[TARGET_LEVEL - 1], LEVEL_ENDS[TARGET_LEVEL]
    for action in full[:start]:
        safe_step(env, action)
    level_groups = groups(full[start:end])
    for keys, pair in level_groups[:TARGET_GROUP]:
        for action in keys + pair:
            safe_step(env, action)
    root = env.clone(); keys, pair = level_groups[TARGET_GROUP]

    baseline = root.clone()
    for action in keys:
        safe_step(baseline, action)

    previous_pair = level_groups[TARGET_GROUP - 1][1]
    compressed = (7,) + runs(keys) + (previous_pair[-1],)
    candidate = root.clone()
    for action in compressed:
        safe_step(candidate, action)
    equal = frame_key(candidate) == frame_key(baseline)
    print("UNDO_SLIDE", TARGET_LEVEL, TARGET_GROUP, keys, runs(keys), len(keys), len(compressed), equal)

    for action in pair:
        safe_step(candidate, action)
    for suffix_keys, suffix_pair in level_groups[TARGET_GROUP + 1:]:
        for action in suffix_keys + suffix_pair:
            safe_step(candidate, action)
        if int(candidate.levels_completed) >= TARGET_LEVEL:
            break
    print("UNDO_SLIDE_SUFFIX", int(candidate.levels_completed), frame_key(candidate) == frame_key(candidate))


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
