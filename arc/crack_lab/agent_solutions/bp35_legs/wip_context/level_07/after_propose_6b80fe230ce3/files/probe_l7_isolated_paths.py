"""Replay short level-7 hypotheses in separate pristine Arena instances."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components


L, R = (3,), (4,)
ROWS = [3 + 6 * i for i in range(10)]
COLS = [15 + 6 * j for j in range(8)]
PALETTE = {0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
           10: ".", 11: "a", 12: "s", 14: "Y", 15: "h"}
CROSS = [
    R, R, R,
    (6, 39, 51),
    (6, 3, 3),
    R,
    (6, 3, 3),
    (7,),
]
VOID = (6, 51, 51)

VARIANTS = {
    "plain": [*CROSS, R],
    "void_only": [*CROSS, VOID, R],
    "void_7": [*CROSS, VOID, (7,), R],
    "void_77": [*CROSS, VOID, (7,), (7,), R],
    "void_twice": [*CROSS, VOID, VOID, R],
    "void_6_7_6": [*CROSS, VOID, (7,), VOID, R],
    "void_7_coord": [*CROSS, VOID, (7, 3, 3), R],
    "void_77_stage": [*CROSS, VOID, (7,), (7,)],
    "void_77_retry": [*CROSS, VOID, (7,), (7,), R, (7,), R],
    "wait1": [*CROSS, (6, 15, 3), R],
    "wait2": [*CROSS, (6, 15, 3), (6, 15, 3), R],
    "wait3": [*CROSS, (6, 15, 3), (6, 15, 3), (6, 15, 3), R],
    "void_777": [*CROSS, VOID, (7,), (7,), (7,)],
    "void_7777": [*CROSS, VOID, (7,), (7,), (7,), (7,)],
    "void_77777": [*CROSS, VOID, (7,), (7,), (7,), (7,), (7,)],
    "void_777777": [*CROSS, VOID, (7,), (7,), (7,), (7,), (7,), (7,)],
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
    return "/".join(
        "".join(PALETTE.get(int(frame[y][x]), "?") for x in COLS)
        for y in ROWS
    )


def zero_pixels(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 0)
    return list(zip(xs.tolist(), ys.tolist()))


def run_variant(name, route):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        for index, action in enumerate(route, 1):
            env.step(*action)
            if name == "void_77_stage":
                print(
                    "ISO_STEP", index, action, env.terminal(),
                    avatar(env.frame()), target(env.frame()),
                    lattice(env.frame()) if index >= 8 else "",
                    zero_pixels(env.frame()) if index >= 8 else [],
                )
            if env.terminal():
                result["stopped"] = index
                break
        result.update(
            level=int(env.levels_completed),
            terminal=bool(env.terminal()),
            avatar=avatar(env.frame()),
            voids=[
                (blob.bbox, blob.area)
                for blob in connected_components(
                    env.frame(), colors=(0,), min_area=2
                )
                if blob.bbox[0] < 63
            ],
        )

    levels, path, error = arena.run_program("bp35", probe)
    print(
        "ISOLATED", name, len(route), result,
        "runner", (levels, len(path), error),
        flush=True,
    )


only = set(filter(None, os.environ.get("ONLY", "").split(",")))
for variant_name, variant_route in VARIANTS.items():
    if not only or variant_name in only:
        run_variant(variant_name, variant_route)
