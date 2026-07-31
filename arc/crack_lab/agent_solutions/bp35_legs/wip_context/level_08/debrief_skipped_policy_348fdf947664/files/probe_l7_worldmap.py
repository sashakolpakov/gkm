"""Reconstruct a compact band-world map along a reproduced safe level-7 route."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, click_action
from perception import connected_components
from probe_level7_reward_recovery import PREFIX, SUFFIX


STATIC = {3: "#", 5: "#", 10: ".", 0: "v"}


def lattice(frame):
    return [
        [STATIC.get(int(frame[y][x])) for x in COL_ANCHORS]
        for y in ROW_ANCHORS
    ]


def avatar_cell(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    y, x = blobs[0].centroid
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def colored_cells(frame, color):
    return [
        (i, j)
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) == color
    ]


def support_cells(frame):
    out = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            if int(frame[y][x]) != 12:
                continue
            r0, c0 = 6 * i, 13 + 6 * j
            area = sum(
                int(frame[r][c]) == 12
                for r in range(r0, min(63, r0 + 6))
                for c in range(c0, c0 + 6)
            )
            out.append((i, j, area))
    return out


def alignment(world, frame_grid, previous):
    candidates = []
    for origin in range(previous - 10, previous + 11):
        matches = mismatches = overlap = 0
        for i, row in enumerate(frame_grid):
            for j, value in enumerate(row):
                known = world.get((origin + i, j))
                if value is None or known is None:
                    continue
                overlap += 1
                if value == known:
                    matches += 1
                else:
                    mismatches += 1
        candidates.append((
            matches - 3 * mismatches,
            matches,
            overlap,
            -abs(origin - previous),
            -abs(origin),
            origin,
        ))
    return max(candidates)


def merge(world, frame_grid, origin):
    for i, row in enumerate(frame_grid):
        for j, value in enumerate(row):
            if value is not None:
                world.setdefault((origin + i, j), value)


def target_cells(frame):
    return [
        (i, j)
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) == 7
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = [
        *PREFIX,
        click_action(5, 2),
        *SUFFIX,
        (3,), (6, 3, 9), (4,), (6, 3, 39),
        (3,), (3,), (3,),
    ]
    world = {}
    origin = 0
    merge(world, lattice(env.frame()), origin)
    print("WORLD_STATE", 0, origin, avatar_cell(env.frame()), [], support_cells(env.frame()))
    for index, action in enumerate(route, 1):
        if env.terminal():
            break
        previous = origin
        env.step(*action)
        if env.terminal():
            print("WORLD_TERMINAL", index, action, env.levels_completed)
            break
        frame = env.frame()
        score = alignment(world, lattice(frame), origin)
        origin = score[-1]
        merge(world, lattice(frame), origin)
        avatar = avatar_cell(frame)
        targets = [(origin + i, j) for i, j in target_cells(frame)]
        supports = [(origin + i, j, area) for i, j, area in support_cells(frame)]
        if origin != previous or targets or action[0] in (6, 7):
            print(
                "WORLD_STATE", index, action, "origin", origin,
                "align", score[:-1],
                "avatar", None if avatar is None else (origin + avatar[0], avatar[1]),
                "targets", targets, "supports", supports,
            )

    rows = range(min(row for row, _ in world), max(row for row, _ in world) + 1)
    for row in rows:
        print("WORLD_ROW", row, "".join(world.get((row, col), "?") for col in range(8)))
    print("WORLD_RESULT", env.levels_completed, env.terminal(), origin)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
