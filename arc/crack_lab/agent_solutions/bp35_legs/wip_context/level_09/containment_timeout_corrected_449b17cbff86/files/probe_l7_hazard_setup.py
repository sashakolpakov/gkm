"""Test the two flanking stage hazards as the omitted setup interaction."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import ROW_ANCHORS, run_actions
from perception import connected_components, frame_delta


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def staged_route():
    with open("frontier_scaffold.json") as stream:
        source = json.load(stream)["staged_prefix_actions"]
    return [
        (action,)
        if isinstance(action, int)
        else (
            (action[0], action[1] + 12, action[2])
            if action[1] != 3
            else tuple(action)
        )
        for action in source
    ]


def avatar(frame):
    pixels = np.asarray(frame)
    ys, xs = np.where(pixels == 9)
    return None if not len(xs) else (round(float(xs.mean())), round(float(ys.mean())))


def hazard_key(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(15,), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    run_actions(env, staged_route())
    base_frame = env.frame()
    hazards = [
        blob
        for blob in connected_components(base_frame, colors=(15,), min_area=2)
        if blob.bbox[0] < 63
    ]
    points = set()
    for blob in hazards:
        r0, c0, r1, c1 = blob.bbox
        points.update(
            (x, y)
            for y in range(r0 - 1, r1 + 2)
            for x in range(c0 - 1, c1 + 2)
        )

    for action in (6, 7):
        for x, y in sorted(points):
            node = env.clone()
            node.step(action, x, y)
            changed_hazard = hazard_key(node.frame()) != hazard_key(base_frame)
            delta = frame_delta(base_frame, node.frame())["count"]
            positions = [avatar(node.frame())]
            for _ in range(4):
                node.step(3)
                positions.append(avatar(node.frame()))
            outcomes = []
            for switch_y in [
                row for row in ROW_ANCHORS if int(node.frame()[row][3]) == 8
            ]:
                branch = node.clone()
                branch.step(6, 3, switch_y)
                outcomes.append(
                    (
                        switch_y,
                        int(branch.levels_completed) - base_level,
                        bool(branch.terminal()),
                        avatar(branch.frame()),
                    )
                )
            if changed_hazard or delta > 1 or any(outcome[1] for outcome in outcomes):
                print(
                    "TRY",
                    (action, x, y),
                    delta,
                    changed_hazard,
                    positions,
                    outcomes,
                )


levels, path, err = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
