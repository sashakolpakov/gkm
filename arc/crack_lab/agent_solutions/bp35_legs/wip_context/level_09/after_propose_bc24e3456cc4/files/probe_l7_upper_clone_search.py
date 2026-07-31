"""Bounded selected-state clone search from the verified upper-right shaft."""

import heapq
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS
from perception import connected_components
from probe_l7_fresh_graph import (
    alignment,
    available_actions,
    lattice,
    merge,
    object_identity,
)
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ENTRY = [
    (7,), (7,),
    (6, 3, 47), (4,), (4,),
]


def extra_actions(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    out = []
    for color in (0, 7, 15):
        for blob in connected_components(frame, colors=(color,), min_area=2):
            if blob.bbox[0] >= 63:
                continue
            y, x = blob.centroid
            if abs(x - ax) <= 18 and abs(y - ay) <= 30:
                out.append((6, round(x), round(y)))
    return out


def cell(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return None
    x, y = avatar
    return (
        min(range(10), key=lambda row: abs(ROW_ANCHORS[row] - y)),
        min(range(8), key=lambda column: abs(COL_ANCHORS[column] - x)),
    )


def priority(world_row, column, distance, depth):
    if os.environ.get("SEARCH_ORDER") == "bfs":
        return depth,
    if distance is not None and distance < 18:
        return 0, distance, depth
    if world_row >= 10:
        return 1, -world_row, depth
    if world_row <= 2:
        return 2, abs(column - 6), depth
    if world_row <= 5:
        return 3, abs(column - 6), depth
    return 4, abs(column - 6), depth


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)

    world = {}
    origin = 0
    merge(world, lattice(env.frame()), origin)
    selected = None
    for action in [*SEED, *ENTRY]:
        before = np.asarray(env.frame()).copy()
        identity = object_identity(before, origin, action)
        env.step(*action)
        if identity is not None:
            selected = identity
        if env.terminal():
            print("ENTRY_TERMINAL")
            return
        origin = alignment(world, lattice(env.frame()), origin)
        merge(world, lattice(env.frame()), origin)

    root_cell = cell(env.frame())
    root_row = origin + root_cell[0]
    counter = itertools.count()
    root = env.clone()
    queue = [
        (
            priority(root_row, root_cell[1], None, 0),
            next(counter),
            root,
            (),
            origin,
            selected,
        )
    ]
    seen = {
        (
            np.asarray(root.frame())[:63].tobytes(),
            origin,
            tuple(selected or ()),
            0,
        )
    }
    best = (18, root_row, root_cell, ())
    max_states = int(os.environ.get("MAX_STATES", "1200"))
    max_depth = int(os.environ.get("MAX_DEPTH", "18"))
    expanded = 0

    while queue and expanded < max_states:
        _, _, node, path, node_origin, node_selected = heapq.heappop(queue)
        if len(path) >= max_depth:
            continue
        frame = np.asarray(node.frame())
        actions = list(
            dict.fromkeys([*available_actions(frame), *extra_actions(frame)])
        )
        for action in actions:
            child = node.clone()
            before = np.asarray(node.frame()).copy()
            identity = object_identity(before, node_origin, action)
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("UPPER_WIN", child_path, expanded, flush=True)
                return
            if child.terminal() or avatar_position(child.frame()) is None:
                if expanded >= max_states:
                    break
                continue

            child_origin = alignment(
                world, lattice(child.frame()), node_origin
            )
            merge(world, lattice(child.frame()), child_origin)
            child_selected = (
                identity if identity is not None else node_selected
            )
            avatar_cell = cell(child.frame())
            world_row = child_origin + avatar_cell[0]
            distance = target_path_distance(child.frame())
            dense = (
                99 if distance is None else distance,
                -world_row,
                avatar_cell,
                child_path,
            )
            if dense < best:
                best = dense
                print(
                    "UPPER_PROGRESS",
                    expanded,
                    "row",
                    world_row,
                    "cell",
                    avatar_cell,
                    "distance",
                    distance,
                    "path",
                    child_path,
                    flush=True,
                )
            if world_row >= 10:
                print(
                    "UPPER_DESCENT",
                    expanded,
                    world_row,
                    avatar_cell,
                    distance,
                    child_path,
                    flush=True,
                )
            key = (
                np.asarray(child.frame())[:63].tobytes(),
                child_origin,
                tuple(child_selected or ()),
                len(child_path) % 6,
            )
            if key not in seen:
                seen.add(key)
                heapq.heappush(
                    queue,
                    (
                        priority(
                            world_row,
                            avatar_cell[1],
                            distance,
                            len(child_path),
                        ),
                        next(counter),
                        child,
                        child_path,
                        child_origin,
                        child_selected,
                    ),
                )
            if expanded >= max_states:
                break
        if expanded and expanded % 100 < len(actions):
            print(
                "UPPER_SEARCH",
                expanded,
                len(queue),
                len(seen),
                best,
                flush=True,
            )
    print("UPPER_DONE", expanded, len(queue), len(seen), best, flush=True)


levels, replay, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(replay), error)
