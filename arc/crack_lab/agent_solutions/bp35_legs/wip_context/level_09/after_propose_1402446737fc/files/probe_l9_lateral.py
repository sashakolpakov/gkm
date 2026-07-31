"""Test lateral catch handoffs at each safe height of the first ladder."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action, moves_used
from perception import connected_components


OPENING = [(6, 3, 39), 4, 4, 4, (6, 3, 5)]
UP = click_action(5, 4)


def enter_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    for action in OPENING:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple(
            (b.color, b.bbox, b.area) for b in blobs if b.bbox[0] < 63
        ),
    }


def probe(env):
    enter_stage(env)
    for height in range(1, 4):
        root = env.clone()
        for _ in range(height):
            root.step(*UP)
        for name, direction, columns in (
            ("left", 3, range(3, -1, -1)),
            ("right", 4, range(5, 8)),
        ):
            child = root.clone()
            print("START", height, name, compact(child))
            for col in columns:
                child.step(*click_action(6, col))
                child.step(direction)
                print("HANDOFF", height, name, col, compact(child))
                if child.terminal():
                    break


arena.run_program("bp35", probe)
