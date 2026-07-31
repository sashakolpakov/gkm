"""Freshly verify the preserved-selection transition from the 38-action seed."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from perception import connected_components
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def summary(env):
    frame = env.frame()
    return {
        "level": int(env.levels_completed),
        "terminal": bool(env.terminal()),
        "avatar": avatar_position(frame),
        "target_distance": target_path_distance(frame),
        "objects": [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                frame, colors=(7, 9, 12, 14, 15), min_area=2
            )
            if blob.bbox[0] < 63
        ],
    }


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    for action in SEED:
        env.step(*action)
    print("SEED", len(SEED), summary(env))
    for count in range(1, int(os.environ.get("RELEASES", "6")) + 1):
        env.step(7)
        print("WAIT", count, summary(env))
        if env.terminal() or env.levels_completed > 6:
            break


levels, path, err = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
