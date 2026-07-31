"""Measure world-coordinate descent of replay-stable level-7 room macros."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, click_action
from probe_level7_reward_recovery import PREFIX, SUFFIX


L, R = (3,), (4,)
TOP = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    L, (6, 3, 9), R, (6, 3, 39),
    L, L, L,
]
DROP = (6, 3, 27)
BRIDGE72 = [
    DROP, (7,), click_action(7, 2), R, (6, 3, 21),
    R, R, R, R,
]
NEXT = [*BRIDGE72, (6, 3, 0)]
STATIC = {3: "#", 5: "#", 10: ".", 0: "v"}

VARIANTS = {
    "top": [],
    "drop": [DROP],
    "drop7": [DROP, (7,)],
    "bridge72": BRIDGE72,
    "bridge72_lefts": [*BRIDGE72, L, L, L, L],
    "bridge72_7_lefts": [*BRIDGE72, (7,), L, L, L, L],
    "next": NEXT,
    "next_cross": [*NEXT, click_action(6, 3), R],
}


def grid(frame):
    return [
        [STATIC.get(int(frame[y][x])) for x in COL_ANCHORS]
        for y in ROW_ANCHORS
    ]


def merge(world, current, origin):
    for i, row in enumerate(current):
        for j, value in enumerate(row):
            if value is not None:
                world.setdefault((origin + i, j), value)


def align(world, current, previous):
    choices = []
    for origin in range(previous - 10, previous + 11):
        matches = mismatches = 0
        for i, row in enumerate(current):
            for j, value in enumerate(row):
                known = world.get((origin + i, j))
                if value is None or known is None:
                    continue
                matches += value == known
                mismatches += value != known
        choices.append((
            matches - 3 * mismatches,
            matches,
            -abs(origin - previous),
            origin,
        ))
    return max(choices)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def run_variant(name, suffix):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        world = {}
        origin = 0
        merge(world, grid(env.frame()), origin)
        for action in [*TOP, *suffix]:
            env.step(*action)
            if env.terminal():
                break
            score = align(world, grid(env.frame()), origin)
            origin = score[-1]
            merge(world, grid(env.frame()), origin)
        cell = None if env.terminal() else avatar(env.frame())
        result.update(
            terminal=bool(env.terminal()),
            level=int(env.levels_completed),
            origin=origin,
            avatar=None if cell is None else (origin + cell[0], cell[1]),
        )

    levels, path, error = arena.run_program("bp35", probe)
    print("ORIGIN", name, result, "runner", (levels, len(path), error), flush=True)


only = set(filter(None, os.environ.get("ONLY", "").split(",")))
for variant_name, variant_suffix in VARIANTS.items():
    if not only or variant_name in only:
        run_variant(variant_name, variant_suffix)
