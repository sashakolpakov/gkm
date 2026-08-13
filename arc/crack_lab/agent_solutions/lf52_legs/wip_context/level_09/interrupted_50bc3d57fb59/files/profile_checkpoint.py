"""Reproduce per-level action counts from the validated checkpoint path."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena


def profile(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    previous_level = int(env.levels_completed)
    previous_boundary = 0
    boundaries = []
    for index, action in enumerate(checkpoint["final_path"], 1):
        env.step(action)
        current_level = int(env.levels_completed)
        if current_level > previous_level:
            boundaries.append((current_level, index - previous_boundary, index))
            previous_level = current_level
            previous_boundary = index
    print("CHECKPOINT_PROFILE", tuple(boundaries))


levels, path, error = arena.run_program("lf52", profile)
print("PROBE_RESULT", levels, len(path), error)
