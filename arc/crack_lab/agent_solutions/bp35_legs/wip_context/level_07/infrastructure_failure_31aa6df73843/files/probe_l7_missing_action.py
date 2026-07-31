"""Recover the one missing action in the 256-move level-7 scaffold witness."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_level7_decode_frontiers import build_route
from probe_level7_decoded_stage import decoded_route


L, R = (3,), (4,)
VARIANTS = {
    "baseline": [L, L, L, L],
    "extra_left": [L, L, L, L, L],
    "extra_right": [L, L, L, L, R],
    "extra_7": [L, L, L, L, (7,)],
    "extra_empty": [L, L, L, L, (6, 15, 3)],
    "extra_strip_top": [L, L, L, L, (6, 3, 3)],
}


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def target(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def lattice(frame):
    rows = [3 + 6 * i for i in range(10)]
    cols = [15 + 6 * j for j in range(8)]
    palette = {
        0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
        10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in cols)
        for y in rows
    )


def switch_components(frame):
    return [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]


with open("frontier_scaffold.json") as _stream:
    _raw_route = json.load(_stream)["staged_prefix_actions"]

ROUTES = {
    "base": decoded_route(),
    "alt": build_route(_raw_route, (False, True, True, True, False)),
}


def run_variant(name, route, suffix):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        for action in [*route, *suffix]:
            env.step(*action)
            if env.terminal() or env.levels_completed > 6:
                break
        switches = [] if env.terminal() else switch_components(env.frame())
        switch_action = None
        if switches and env.levels_completed == 6:
            blob = max(switches, key=lambda item: item.centroid[0])
            y, x = blob.centroid
            switch_action = (6, round(x), round(y))
            env.step(*switch_action)
        result.update(
            level=int(env.levels_completed),
            terminal=bool(env.terminal()),
            avatar=avatar(env.frame()),
            target=target(env.frame()),
            lattice=lattice(env.frame()),
            switch=switch_action,
        )

    levels, path, error = arena.run_program("bp35", probe)
    print(
        "MISSING", name, len(suffix) + 1, result,
        "runner", (levels, len(path), error), flush=True,
    )


only = set(filter(None, os.environ.get("ONLY", "").split(",")))
for route_name, route_actions in ROUTES.items():
    for variant_name, variant_suffix in VARIANTS.items():
        full_name = f"{route_name}_{variant_name}"
        if not only or full_name in only:
            run_variant(full_name, route_actions, variant_suffix)
