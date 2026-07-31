"""Descend the staged catch through the central gap one band at a time."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action, moves_used
from perception import connected_components


def enter_down_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    route = [
        (6, 3, 39), 4, 4, 4, (6, 3, 5),
        click_action(5, 4),
        click_action(1, 2),
        click_action(6, 3), 3,
        click_action(6, 2), 3,
        *([click_action(5, 2)] * 5),
        (6, 3, 9),
        4,
    ]
    for action in route:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def switches(env):
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(8,), min_area=3)
        if blob.bbox[0] < 63
    ]


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


def probe(env):
    enter_down_stage(env)
    node = env.clone()
    print("START", compact(node))
    for advance in range(1, 11):
        if node.terminal():
            break
        node.step(*click_action(5, 3))
        print("DESCEND", advance, compact(node))
        for switch in switches(node):
            child = node.clone()
            child.step(*switch)
            print("FLIP", advance, switch, compact(child))


arena.run_program("bp35", probe)
