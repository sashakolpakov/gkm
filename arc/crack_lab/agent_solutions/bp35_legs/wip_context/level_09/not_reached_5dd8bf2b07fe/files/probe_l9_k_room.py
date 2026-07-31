"""Enumerate the unique column-5 upper transition room."""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, band_grid, click_action, moves_used
from perception import color_counts, connected_components
from probe_l9_top_handoff import enter_top


def enter_k_room(env):
    enter_top(env)
    env.step(*click_action(3, 2))
    for col in (6, 5):
        env.step(*click_action(6, col))
        env.step(3)
    env.step(*click_action(5, 5))


def compact(env):
    blobs = connected_components(env.frame(), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "colors": color_counts(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color not in (5, 10)
        ),
    }


def signature(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def stepped(root, action):
    child = root.clone()
    child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_k_room(env)
    print("K_ROOM", compact(env))
    for action in (3, 4, 7):
        print("KEY", action, compact(stepped(env, action)))

    points = [
        (6, x, y)
        for y in ROW_ANCHORS
        for x in COL_ANCHORS
    ]
    points.extend(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(7, 8, 12, 14, 15), min_area=3)
        if blob.bbox[0] < 63
    )
    groups = defaultdict(list)
    representatives = {}
    for action in dict.fromkeys(points):
        child = stepped(env, action)
        key = signature(child)
        groups[key].append(action)
        representatives[key] = compact(child)
    for actions, result in sorted(
        ((groups[key], representatives[key]) for key in groups),
        key=lambda item: (len(item[0]), item[0]),
    ):
        print("CLICK", actions, result)

    child = env.clone()
    for advance in range(1, 5):
        if child.terminal():
            break
        child.step(*click_action(5, 5))
        print("ADVANCE", advance, compact(child))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
