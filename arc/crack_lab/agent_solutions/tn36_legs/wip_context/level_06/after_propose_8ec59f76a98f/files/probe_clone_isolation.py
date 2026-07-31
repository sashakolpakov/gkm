"""Verify that coordinate actions on a clone do not alter its level-4 parent."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, frame_delta


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    parent_before = arr(env.frame()).copy()
    first = env.clone()
    first.step(6, 34, 33)
    print(
        "after_segment",
        {
            "parent": frame_delta(parent_before, env.frame()),
            "clone": frame_delta(parent_before, first.frame()),
            "levels": (env.levels_completed, first.levels_completed),
        },
    )

    second = env.clone()
    print(
        "second_entry",
        {
            "parent": frame_delta(parent_before, env.frame()),
            "second": frame_delta(parent_before, second.frame()),
        },
    )

    first.step(6, 57, 58)
    print(
        "after_submit",
        {
            "parent": frame_delta(parent_before, env.frame()),
            "levels": (env.levels_completed, first.levels_completed),
        },
    )
    third = env.clone()
    print(
        "third_entry",
        {
            "parent": frame_delta(parent_before, env.frame()),
            "third": frame_delta(parent_before, third.frame()),
            "levels": (env.levels_completed, third.levels_completed),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
