"""Compact fresh observations at the level-7 entry."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from perception import connected_components, frame_delta


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def objects(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
        if blob.bbox[0] < 63 and blob.area < 1000
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    base = env.frame()
    print("ENTRY", int(env.levels_completed), repr(env.actions), objects(base))
    for action in (3, 4, 7):
        node = env.clone()
        node.step(action)
        print(
            "KEY",
            action,
            frame_delta(base, node.frame()),
            int(node.levels_completed),
            bool(node.terminal()),
            objects(node.frame()),
        )

    for action in (6, 7):
        changed = []
        for y in range(3, 58, 6):
            for x in (3, *range(15, 58, 6)):
                node = env.clone()
                node.step(action, x, y)
                delta = frame_delta(base, node.frame())
                if delta["count"] > 1:
                    changed.append(
                        ((y, x), delta["count"], delta["bbox"], bool(node.terminal()))
                    )
        print("COORD", action, changed)


arena.run_program("bp35", probe)
