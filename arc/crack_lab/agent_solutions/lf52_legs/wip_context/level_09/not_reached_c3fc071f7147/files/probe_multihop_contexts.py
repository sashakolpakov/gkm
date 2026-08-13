"""Test far destinations only where a verified relay chain is present."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def pegs(frame):
    return tuple(sorted(
        blob.top_left for blob in connected_components(frame, colors=(14,))
        if blob.size == (4, 4)
    ))


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    # Level 5, after the first main-board bridge has aligned at column 15.
    for action in campaign[:192]:
        safe_step(env, action)
    outcomes = []
    for destination_col in (21, 33, 45):
        child = env.clone()
        before = child.frame()
        safe_step(child, (6, 10, 25))
        safe_step(child, (6, destination_col + 1, 25))
        outcomes.append((destination_col, delta(before, child.frame()),
                         pegs(child.frame())))
    print("l5_multihop", pegs(env.frame()), tuple(outcomes), flush=True)


arena.run_program("lf52", probe)
