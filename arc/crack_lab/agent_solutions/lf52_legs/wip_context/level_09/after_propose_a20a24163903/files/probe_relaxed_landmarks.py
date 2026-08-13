"""Shorten key runs while preserving successful coordinate macros and reward."""

import json
import os

import gkm_try
from legs import _bridge_carrier_state
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


def shorten(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start]:
        env.step(action)
    entry = env.clone()
    validation_root = env.clone()
    segment = full_path[start:end]

    groups = []
    index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not is_coordinate(segment[index]):
            keys.append(segment[index])
            index += 1
        clicks = []
        while index < len(segment) and is_coordinate(segment[index]) and len(clicks) < 2:
            clicks.append(segment[index])
            index += 1
        groups.append((keys, clicks))

    node = entry.clone()
    optimized = []
    sizes = []
    trials = 0
    stalled = None

    def try_group(key_run, clicks):
        nonlocal trials
        trials += 1
        child = node.clone()
        for action in key_run:
            child.step(action)
        before_clicks = arr(child.frame())[1:, :].tobytes()
        for action in clicks:
            child.step(action)
        after_clicks = arr(child.frame())[1:, :].tobytes()
        selected = _bridge_carrier_state(child.frame())[5]
        succeeded = (
            child.levels_completed >= TARGET_LEVEL
            or (after_clicks != before_clicks and selected is None)
        )
        return succeeded, child

    for group_index, (original_run, clicks) in enumerate(groups):
        run = list(original_run)
        changed = True
        while changed:
            changed = False
            for remove_index in range(len(run)):
                trial_run = run[:remove_index] + run[remove_index + 1:]
                succeeded, _ = try_group(trial_run, clicks)
                if succeeded:
                    run = trial_run
                    changed = True
                    break
        succeeded, child = try_group(run, clicks)
        if not succeeded:
            succeeded, child = try_group(original_run, clicks)
            run = list(original_run)
        if not succeeded:
            stalled = group_index
            break
        node = child
        optimized.extend(run + clicks)
        sizes.append((len(original_run), len(run)))
        if node.levels_completed >= TARGET_LEVEL:
            break

    valid = False
    if stalled is None:
        for action in optimized:
            validation_root.step(action)
        valid = validation_root.levels_completed >= TARGET_LEVEL
    print(
        "RELAXED_RESULT", TARGET_LEVEL, len(segment), len(optimized),
        trials, stalled, valid, sizes,
    )
    print("RELAXED_PATH", optimized)


gkm_try.A.run_program("lf52", shorten)
