"""Click the actual hanging hazard pixels before entering the target shaft."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from perception import connected_components, frame_delta
from probe_level7_reward_recovery import avatar_cell, controls, lattice


RIGHT, LEFT = (4,), (3,)
ENTRY = [
    click_action(2, 2),
    click_action(4, 2),
    click_action(4, 4),
    click_action(1, 3),
    RIGHT, RIGHT, RIGHT,
    click_action(8, 4),
    (6, 3, 3),
    RIGHT,
]


def hazards(frame):
    return [
        blob
        for blob in connected_components(frame, colors=(15,), min_area=2)
        if blob.bbox[0] < 63
    ]


def summary(node):
    return (
        int(node.levels_completed),
        bool(node.terminal()),
        None if node.terminal() else avatar_cell(node.frame()),
        () if node.terminal() else tuple(
            (blob.bbox, blob.area) for blob in hazards(node.frame())
        ),
        () if node.terminal() else tuple(controls(node.frame())),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)
    for action in ENTRY:
        env.step(*action)
        if env.terminal():
            print("SHAFT_HAZARD_ENTRY_DEAD", action)
            return
    root = env.clone()
    base = root.frame()
    print("SHAFT_HAZARD_ROOT", summary(root), flush=True)

    candidates = []
    for blob in hazards(base):
        r0, c0, r1, c1 = blob.bbox
        candidates.extend(
            [
                (6, round(blob.centroid[1]), round(blob.centroid[0])),
                (6, c0, r0),
                (6, c1, r1),
                (7, round(blob.centroid[1]), round(blob.centroid[0])),
            ]
        )
    for action in dict.fromkeys(candidates):
        node = root.clone()
        before = node.frame()
        node.step(*action)
        delta = frame_delta(before, node.frame())
        after_click = summary(node)
        path = [action]
        for move in [RIGHT, RIGHT, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT]:
            if node.terminal() or node.levels_completed > base_level:
                break
            node.step(*move)
            path.append(move)
        print(
            "SHAFT_HAZARD_TRY", action,
            (delta["count"], delta["bbox"]),
            after_click, "PATH", path, "END", summary(node), flush=True,
        )
        if node.levels_completed > base_level:
            print("SHAFT_HAZARD_WIN", [*ENTRY, *path], flush=True)
            return


arena.run_program("bp35", probe)
