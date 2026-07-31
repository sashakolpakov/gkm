"""One-action falsification at the reviewed level-7 staged frame."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS, run_actions
from perception import connected_components


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar_xy(frame):
    pixels = np.asarray(frame)
    ys, xs = np.where(pixels == 9)
    if not len(xs):
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def targets(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    ]


def distance(frame):
    avatar = avatar_xy(frame)
    blobs = targets(frame)
    if avatar is None or not blobs:
        return 999
    ax, ay = avatar
    return min(
        abs(ax - round(blob.centroid[1])) + abs(ay - round(blob.centroid[0]))
        for blob in blobs
    )


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


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    run_actions(env, staged_route())

    print(
        "STAGE",
        avatar_xy(env.frame()),
        targets(env.frame()),
        [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                env.frame(), colors=(12, 14, 15), min_area=2
            )
            if blob.bbox[0] < 63
        ],
    )
    candidates = [(3,), (4,), (7,)]
    candidates += [(6, 3, y) for y in ROW_ANCHORS]
    candidates += [
        (6, x, y) for y in ROW_ANCHORS for x in COL_ANCHORS
    ]
    candidates += [
        (7, x, y) for y in ROW_ANCHORS for x in COL_ANCHORS
    ]

    best = []
    for candidate in candidates:
        node = env.clone()
        node.step(*candidate)
        positions = [avatar_xy(node.frame())]
        for _ in range(4):
            node.step(3)
            positions.append(avatar_xy(node.frame()))
        switches = [
            y for y in ROW_ANCHORS if int(node.frame()[y][3]) == 8
        ]
        outcomes = []
        for y in switches:
            branch = node.clone()
            branch.step(6, 3, y)
            outcomes.append(
                (
                    y,
                    int(branch.levels_completed) - base_level,
                    bool(branch.terminal()),
                    avatar_xy(branch.frame()),
                    distance(branch.frame()),
                )
            )
        score = min(
            [distance(node.frame())]
            + [outcome[-1] for outcome in outcomes]
        )
        if (
            score < 12
            or len(set(positions)) > 1
            or any(outcome[1] for outcome in outcomes)
        ):
            best.append((score, candidate, positions, outcomes))
    for result in sorted(best, key=lambda item: (item[0], item[1])):
        print("BRANCH", result)


levels, path, err = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
