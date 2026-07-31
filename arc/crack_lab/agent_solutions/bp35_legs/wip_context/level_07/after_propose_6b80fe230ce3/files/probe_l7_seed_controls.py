"""Compact control-action probes from the reproduced 38-step level-7 state."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_grid, band_shift, run_actions
from perception import color_counts, connected_components, frame_delta
from probe_l7_raw_search import SEED, avatar_position


def compact(frame):
    controls = [
        (blob.bbox, blob.area, tuple(round(value, 1) for value in blob.centroid))
        for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63
    ]
    return {
        "avatar": avatar_position(frame),
        "colors": color_counts(frame),
        "corners": [int(frame[y][x]) for y, x in ((0, 0), (0, 3), (3, 3))],
        "controls": controls,
        "grid": [list(row) for row in band_grid(frame)],
    }


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    print("INITIAL", compact(env.frame()))
    run_actions(env, SEED)
    print("SEED", compact(env.frame()))

    targets = {(0, 0)}
    for blob in connected_components(env.frame(), colors=(8,), min_area=2):
        if blob.bbox[0] >= 63:
            continue
        y, x = blob.centroid
        targets.add((int(round(x)), int(round(y))))
    for kind in (6, 7):
        for x, y in sorted(targets):
            node = env.clone()
            before = np.asarray(node.frame()).copy()
            node.step(kind, x, y)
            after = np.asarray(node.frame())
            print(
                "TRY",
                (kind, x, y),
                {
                    "terminal": bool(node.terminal()),
                    "avatar": avatar_position(after),
                    "rise": band_shift(before, after),
                    "delta": frame_delta(before[:63], after[:63]),
                    "state": compact(after),
                },
            )


arena.run_program("bp35", probe)
