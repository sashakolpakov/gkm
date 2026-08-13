"""Find shorter key alignments that preserve the complete level-7 suffix."""

from collections import deque
import json
import os

import gkm_try

from legs import _movable_bridge_board
from perception import arr, safe_step
from probe_undo_slide import groups


LEVEL_START = 316
LEVEL_END = 461
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "13"))
MAX_STATES = int(os.environ.get("MAX_STATES", "1800"))
MIN_SAVE = int(os.environ.get("MIN_SAVE", "3"))
MAX_VALIDATIONS = int(os.environ.get("MAX_VALIDATIONS", "160"))


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    _, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return carriers, bridges, pegs


def apply_all(node, actions, stop_level=None):
    for action in actions:
        safe_step(node, action)
        if stop_level is not None and int(node.levels_completed) >= stop_level:
            break


def probe(env):
    with open("optimized_prefix_l4_l6_candidate.json") as candidate_file:
        full = json.load(candidate_file)["final_path"]
    apply_all(env, full[:LEVEL_START])
    level_groups = list(groups(full[LEVEL_START:LEVEL_END]))
    # A suffix-aware replay independently verified this one-key deletion.
    keys, pair = level_groups[21]
    level_groups[21] = (keys[:6] + keys[7:], pair)

    for keys, pair in level_groups[:TARGET_GROUP]:
        apply_all(env, keys + pair)
    root = env.clone()
    original_keys, pair = level_groups[TARGET_GROUP]
    suffix = tuple(
        action
        for keys, coordinate_pair in level_groups[TARGET_GROUP + 1:]
        for action in keys + coordinate_pair
    )

    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    candidates = {}
    legal = 0
    found = None
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        before = compact(node)
        moved = node.clone(); apply_all(moved, pair)
        if compact(moved) != before or int(moved.levels_completed) >= 7:
            legal += 1
            post_key = frame_key(moved)
            old = candidates.get(post_key)
            if old is None or len(path) < len(old[0]):
                candidates[post_key] = path, moved
        if len(path) >= len(original_keys) - MIN_SAVE:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (action,)))
    validations = 0
    for path, moved in sorted(candidates.values(), key=lambda item: (len(item[0]), item[0])):
        if validations >= MAX_VALIDATIONS:
            break
        validations += 1
        validation = moved.clone(); apply_all(validation, suffix, stop_level=7)
        if int(validation.levels_completed) >= 7:
            found = path
            break
    print(
        "L7_SUFFIX_FRONTIER", TARGET_GROUP, len(original_keys), len(seen),
        legal, len(candidates), validations, found,
    )


gkm_try.A.run_program("lf52", probe)
