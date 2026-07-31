"""Verify observational isolation of sibling level-6 clones."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from perception import arr, frame_delta


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    root_frame = arr(env.frame()).copy()
    baseline = env.clone()
    baseline.step(6, 34, 33)
    baseline_frame = arr(baseline.frame()).copy()

    disturb = env.clone()
    for col in (34, 39, 44, 49, 54, 59):
        disturb.step(6, col, 36)
    disturb.step(6, 57, 58)

    repeated = env.clone()
    repeated.step(6, 34, 33)
    print(
        "isolation",
        {
            "root_after": frame_delta(root_frame, env.frame()),
            "baseline": frame_delta(root_frame, baseline_frame),
            "repeated": frame_delta(root_frame, repeated.frame()),
            "same_outcome": frame_delta(baseline_frame, repeated.frame()),
            "levels": (
                env.levels_completed,
                baseline.levels_completed,
                disturb.levels_completed,
                repeated.levels_completed,
            ),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
