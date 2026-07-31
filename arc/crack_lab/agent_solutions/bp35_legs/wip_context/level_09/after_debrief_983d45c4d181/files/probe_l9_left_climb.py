"""Carry the catch ladder into each removable left lane and climb."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action, moves_used
from perception import connected_components


OPENING = [(6, 3, 39), 4, 4, 4, (6, 3, 5), click_action(5, 4)]


def enter_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    for action in OPENING:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 14), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple(
            (b.color, b.bbox, b.area) for b in blobs if b.bbox[0] < 63
        ),
    }


def handoff_left(env, target):
    for col in range(3, target - 1, -1):
        env.step(*click_action(6, col))
        env.step(3)


def probe(env):
    enter_stage(env)
    for target in range(3):
        for preclear in (False, True):
            child = env.clone()
            if preclear:
                child.step(*click_action(1, target))
            handoff_left(child, target)
            print("LANE", target, preclear, compact(child))
            for advance in range(1, 9):
                if child.terminal():
                    break
                child.step(*click_action(5, target))
                print("CLIMB", target, preclear, advance, compact(child))


arena.run_program("bp35", probe)
