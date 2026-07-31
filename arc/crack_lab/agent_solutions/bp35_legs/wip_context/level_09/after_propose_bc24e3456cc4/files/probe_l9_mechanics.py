"""Reproduce level-9 movement, control, undo, and propagation mechanics."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import band_grid, moves_used
from perception import connected_components


TARGET = (6, 39, 15)
UPPER_SWITCH = (6, 3, 27)
LOWER_SWITCH = (6, 3, 39)


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def compact(env):
    frame = env.frame()
    blobs = connected_components(frame, colors=(8, 9, 11, 15), min_area=2)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(frame),
        "grid": tuple("".join(row) for row in band_grid(frame)),
        "pieces": tuple((b.color, b.bbox, b.area) for b in blobs if b.bbox[0] < 63),
    }


def run(root, name, actions):
    env = root.clone()
    print("PATH", name, "START", compact(env))
    for index, action in enumerate(actions, 1):
        if env.terminal():
            break
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        print("STEP", name, index, action, compact(env))


def probe(env):
    enter_level_9(env)
    tests = {
        "right4": [4, 4, 4, 4],
        "left4": [3, 3, 3, 3],
        "undo1": [4, 7],
        "undo2": [4, 4, 7, 7],
        "upper": [UPPER_SWITCH, 4, 4, 4],
        "lower": [LOWER_SWITCH, 4, 4, 4],
        "split": [TARGET],
        "split_up": [TARGET, (6, 39, 9)],
        "split_left": [TARGET, (6, 33, 15)],
        "split_down": [TARGET, (6, 39, 21)],
        "split_right": [TARGET, (6, 45, 15)],
    }
    for name, actions in tests.items():
        run(env, name, actions)


arena.run_program("bp35", probe)
