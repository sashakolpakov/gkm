"""Minimize key runs between validated coordinate-move landmarks."""

import json
import os
from collections import deque

import gkm_try
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
SEARCH_CAP = int(os.environ.get("SEARCH_CAP", "300"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "-1"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


def optimize(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]

    start_index = LEVEL_ENDS[TARGET_LEVEL - 1]
    end_index = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start_index]:
        env.step(action)
    entry = env.clone()
    segment = full_path[start_index:end_index]

    landmarks = []
    index = 0
    original = entry.clone()
    while index < len(segment):
        key_run = []
        while index < len(segment) and not is_coordinate(segment[index]):
            key_run.append(segment[index])
            original.step(segment[index])
            index += 1
        pre_key = (
            original.levels_completed,
            arr(original.frame())[1:, :].tobytes(),
        )
        pre_node = original.clone()
        coordinate_pair = []
        while index < len(segment) and is_coordinate(segment[index]) and len(coordinate_pair) < 2:
            coordinate_pair.append(segment[index])
            original.step(segment[index])
            index += 1
        landmarks.append((key_run, coordinate_pair, pre_key, pre_node, (
            original.levels_completed,
            arr(original.frame())[1:, :].tobytes(),
        )))

    node = entry.clone()
    optimized = []
    trials = 0
    run_sizes = []
    searched_states = 0
    for group_index, (original_run, coordinate_pair, pre_key, pre_node, target_key) in enumerate(landmarks):
        run = list(original_run)
        if TARGET_GROUP >= 0 and group_index != TARGET_GROUP:
            for action in run + coordinate_pair:
                node.step(action)
            optimized.extend(run + coordinate_pair)
            run_sizes.append((len(original_run), len(run)))
            continue
        changed = True
        while changed:
            changed = False
            for remove_index in range(len(run)):
                trial_run = run[:remove_index] + run[remove_index + 1:]
                child = node.clone()
                for action in trial_run + coordinate_pair:
                    child.step(action)
                trials += 1
                child_key = (
                    child.levels_completed,
                    arr(child.frame())[1:, :].tobytes(),
                )
                if child_key == target_key:
                    run = trial_run
                    changed = True
                    break
        if run:
            def state_key(search_node):
                return (
                    search_node.levels_completed,
                    arr(search_node.frame())[1:, :].tobytes(),
                )

            def paths_from(root, max_depth):
                root_key = state_key(root)
                paths = {root_key: ()}
                queue = deque([()])
                while queue and len(paths) <= SEARCH_CAP:
                    path = queue.popleft()
                    if len(path) >= max_depth:
                        continue
                    for action in (1, 2, 3, 4):
                        child_path = path + (action,)
                        child = root.clone()
                        for replay_action in child_path:
                            child.step(replay_action)
                        child_key = state_key(child)
                        if child_key in paths:
                            continue
                        paths[child_key] = child_path
                        queue.append(child_path)
                return paths

            search_depth = (len(run) - 1 + 1) // 2
            forward = paths_from(node, search_depth)
            backward = paths_from(pre_node, search_depth)
            searched_states += len(forward) + len(backward)
            inverse = {1: 2, 2: 1, 3: 4, 4: 3}
            candidates = []
            for meeting in forward.keys() & backward.keys():
                suffix = tuple(inverse[action] for action in reversed(backward[meeting]))
                candidate = forward[meeting] + suffix
                if len(candidate) < len(run):
                    candidates.append(candidate)
            for candidate in sorted(candidates, key=len):
                child = node.clone()
                for action in candidate:
                    child.step(action)
                if state_key(child) == pre_key:
                    run = list(candidate)
                    break
        for action in run + coordinate_pair:
            node.step(action)
        optimized.extend(run + coordinate_pair)
        run_sizes.append((len(original_run), len(run)))

    validation = entry.clone()
    for action in optimized:
        validation.step(action)
    print(
        "OPT_RESULT", TARGET_LEVEL, len(segment), len(optimized), trials,
        searched_states, validation.levels_completed >= TARGET_LEVEL, run_sizes,
    )
    print("OPT_PATH", optimized)


gkm_try.A.run_program("lf52", optimize)
