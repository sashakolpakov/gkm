"""Test the one omitted opening support in the 256-move level-7 witness."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from perception import connected_components
from probe_level7_decoded_stage import decoded_route


L = (3,)
EXISTING = {(2, 2), (4, 2), (4, 4), (1, 3)}
CANDIDATES = [
    (row, column)
    for row in range(5)
    for column in range(2, 5)
    if (row, column) not in EXISTING
] + [(6, 4), (8, 4)]


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def run_candidate(cell):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        base = decoded_route()
        route = [
            *base[:4],
            click_action(*cell),
            *base[4:],
            L, L, L, L,
        ]
        for action in route:
            env.step(*action)
            if env.terminal() or env.levels_completed > 6:
                break
        switches = [
            blob for blob in connected_components(
                env.frame(), colors=(8,), min_area=2
            )
            if blob.bbox[0] < 63 and blob.bbox[1] <= 5
        ]
        switch = None
        if switches and env.levels_completed == 6 and not env.terminal():
            blob = max(switches, key=lambda item: item.centroid[0])
            y, x = blob.centroid
            switch = (6, round(x), round(y))
            env.step(*switch)
        result.update(
            level=int(env.levels_completed),
            terminal=bool(env.terminal()),
            avatar=avatar(env.frame()),
            switch=switch,
        )

    levels, path, error = arena.run_program("bp35", probe)
    print(
        "INITIAL_MISSING", cell, result,
        "runner", (levels, len(path), error), flush=True,
    )
    return result["level"] > 6


for candidate in CANDIDATES:
    if run_candidate(candidate):
        print("INITIAL_WIN", candidate, flush=True)
        break
