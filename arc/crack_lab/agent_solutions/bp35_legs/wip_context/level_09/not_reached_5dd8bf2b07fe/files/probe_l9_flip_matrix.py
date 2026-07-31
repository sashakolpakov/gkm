"""Map two-switch gravity round trips from each reachable platform column."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, moves_used
from perception import connected_components


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def switches(env):
    blobs = connected_components(env.frame(), colors=(8,), min_area=3)
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in blobs
        if blob.bbox[0] < 63
    ]


def compact(env):
    blobs = connected_components(env.frame(), colors=(8, 9, 11, 15), min_area=2)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple((b.color, b.bbox, b.area) for b in blobs if b.bbox[0] < 63),
    }


def probe(env):
    enter_level_9(env)
    for first in switches(env):
        for rights in range(7):
            child = env.clone()
            child.step(*first)
            for _ in range(rights):
                if not child.terminal():
                    child.step(4)
            remaining = switches(child)
            if remaining and not child.terminal():
                child.step(*remaining[0])
            print("ROUNDTRIP", first, rights, remaining, compact(child))


arena.run_program("bp35", probe)
