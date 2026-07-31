"""Stage a near downward catch before leaving the upper-left room."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, click_action, moves_used
from perception import connected_components


def enter_upper(env, clears):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    route = [
        (6, 3, 39), 4, 4, 4, (6, 3, 5),
        click_action(5, 4),
        *[click_action(1, col) for col in clears],
        click_action(6, 3), 3,
        click_action(6, 2), 3,
        *([click_action(5, 2)] * 5),
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
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "switches": switches(env),
        "pieces": tuple(
            (b.color, b.bbox, b.area) for b in blobs
            if b.bbox[0] < 63 and b.color in (7, 8, 9, 14)
        ),
    }


def probe(env):
    for clears in ((2,), (1, 2), (0, 1, 2)):
        root = env.clone()
        enter_upper(root, clears)
        for staged in ((7, 1), (7, 3), (7, 1, 7, 3)):
            node = root.clone()
            for index in range(0, len(staged), 2):
                node.step(*click_action(staged[index], staged[index + 1]))
            print("STAGED", clears, staged, compact(node))
            for switch in switches(node):
                child = node.clone()
                child.step(*switch)
                print("FLIP", clears, staged, switch, compact(child))
                for action in (3, 4):
                    moved = child.clone()
                    moved.step(action)
                    print("MOVE", clears, staged, switch, action, compact(moved))


arena.run_program("bp35", probe)
