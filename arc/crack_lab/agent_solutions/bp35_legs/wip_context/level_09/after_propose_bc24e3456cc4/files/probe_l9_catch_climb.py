"""Trace repeated catch-ladder advances and switch timing at stage 1."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action, moves_used
from perception import connected_components


OPENING = [(6, 3, 39), 4, 4, 4, (6, 3, 5)]
CATCH = click_action(5, 4)


def enter_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    for action in OPENING:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def switches(env):
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(8,), min_area=3)
        if blob.bbox[0] < 63
    ]


def compact(env):
    blobs = connected_components(env.frame(), colors=(9, 11, 14, 15), min_area=2)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "switches": switches(env),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color in (9, 14, 15)
        ),
    }


def probe(env):
    enter_stage(env)
    node = env.clone()
    for advance in range(1, 9):
        if node.terminal():
            break
        node.step(*CATCH)
        print("ADVANCE", advance, compact(node))
        for action in (3, 4, *switches(node)):
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            print("TRY", advance, action, compact(child))


arena.run_program("bp35", probe)
