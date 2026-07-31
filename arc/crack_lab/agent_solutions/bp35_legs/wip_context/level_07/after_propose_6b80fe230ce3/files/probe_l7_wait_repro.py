"""Reproduce the reviewed level-7 route with its missing wait action."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import ROW_ANCHORS, run_actions
from perception import connected_components


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    return blobs[0].bbox if blobs else None


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    with open("frontier_scaffold.json") as stream:
        staged_prefix = json.load(stream)["staged_prefix_actions"]
    for action in prefix:
        env.step(action)

    base_level = int(env.levels_completed)
    route = [
        (action,)
        if isinstance(action, int)
        else (
            (action[0], action[1] + 12, action[2])
            if action[1] != 3
            else tuple(action)
        )
        for action in staged_prefix
    ]
    route.extend([(7,), (3,), (3,), (3,), (3,)])
    run_actions(env, route)
    switches = [
        y for y in ROW_ANCHORS if int(env.frame()[y][3]) == 8
    ]
    print(
        "BEFORE_FINAL",
        len(route),
        int(env.levels_completed) - base_level,
        bool(env.terminal()),
        avatar(env.frame()),
        switches,
    )
    if switches:
        env.step(6, 3, max(switches))
    print(
        "AFTER_FINAL",
        len(route) + 1,
        int(env.levels_completed) - base_level,
        bool(env.terminal()),
        avatar(env.frame()),
    )


levels, path, err = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
