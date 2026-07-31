"""Verify level-7 sibling clones, including selector-state isolation."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import learn_direction_protocol_from_selector
from perception import arr, connected_components, frame_delta


def board_objects(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=range(7, 16), min_area=2)
        if blob.bbox[2] < 32 and blob.bbox[1] >= 32
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    root_frame = arr(env.frame()).copy()
    baseline = env.clone()
    baseline.step(6, 59, 33)
    baseline_frame = arr(baseline.frame()).copy()

    disturb = env.clone()
    learned = learn_direction_protocol_from_selector(disturb)
    disturb.step(6, 57, 58)

    repeated = env.clone()
    repeated.step(6, 59, 33)
    repeated_frame = arr(repeated.frame()).copy()

    print(
        "isolation",
        {
            "learned": learned,
            "root_after": frame_delta(root_frame, env.frame()),
            "baseline": frame_delta(root_frame, baseline_frame),
            "repeated": frame_delta(root_frame, repeated_frame),
            "same_outcome": frame_delta(baseline_frame, repeated_frame),
            "baseline_objects": board_objects(baseline_frame),
            "repeated_objects": board_objects(repeated_frame),
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
