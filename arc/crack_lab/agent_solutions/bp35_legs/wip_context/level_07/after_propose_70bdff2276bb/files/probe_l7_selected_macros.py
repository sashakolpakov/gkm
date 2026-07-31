"""Room-macro beam search retaining the remotely selected object."""

from collections import deque
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions,
)
from perception import connected_components
from probe_l7_frontier import BASE
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


STATIC = {3: "#", 5: "#", 10: ".", 0: "v"}
GATE_ROOT = [
    (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
    (4,), (4,), (4,), (6, 39, 51), (6, 3, 3), (4,),
]


def lattice(frame):
    return [
        [STATIC.get(int(frame[y][x])) for x in COL_ANCHORS]
        for y in ROW_ANCHORS
    ]


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
            matches - 3 * mismatches, matches, overlap,
            -abs(origin - previous), -abs(origin), origin,
        ))
    return max(candidates)


def merge(world, frame_grid, origin):
    for i, row in enumerate(frame_grid):
        for j, value in enumerate(row):
            if value is not None:
                world.setdefault((origin + i, j), value)


def token_after(frame, action):
    return (
        hashlib.blake2b(
            np.asarray(frame)[:63].tobytes(), digest_size=8
        ).digest(),
        action,
    )


def action_choices(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - ay))
    aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - ax))
    out = [(3,), (4,), (7,)]
    crosses = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    crosses.sort(key=lambda blob: abs(blob.centroid[0] - ay))
    for blob in crosses[:2]:
        y, x = blob.centroid
        out.append((6, int(round(x)), int(round(y))))
    for i in range(max(0, ai - 3), min(10, ai + 4)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            if _cell_shape(frame, i, j)[0] in (12, 14):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def frame_key(frame, token, origin):
    return np.asarray(frame)[:63].tobytes(), token, origin


def local_successors(
    root, root_token, root_origin, world, base_level, max_local=45
):
    queue = deque([(root, (), root_token, root_origin)])
    seen = {frame_key(root.frame(), root_token, root_origin)}
    outcomes = {}
    expanded = 0
    while queue and expanded < max_local:
        node, path, token, origin = queue.popleft()
        expanded += 1
        if len(path) >= 7:
            continue
        before = np.asarray(node.frame()).copy()
        for action in action_choices(before):
            child = node.clone()
            child.step(*action)
            child_path = (*path, action)
            child_token = token_after(before, action) if action[0] == 6 else token
            if child.levels_completed > base_level:
                return [], child_path, expanded
            if child.terminal() or avatar_position(child.frame()) is None:
                continue
            after = np.asarray(child.frame())
            child_origin = alignment(world, lattice(after), origin)[-1]
            merge(world, lattice(after), child_origin)
            changed = int((before[:63] != after[:63]).sum())
            key = frame_key(after, child_token, child_origin)
            if key in seen:
                continue
            seen.add(key)
            if changed >= 400:
                current = outcomes.get(key)
                if current is None or len(child_path) < len(current[0]):
                    outcomes[key] = (
                        child_path, child_token, child, child_origin
                    )
            else:
                queue.append(
                    (child, child_path, child_token, child_origin)
                )
    return list(outcomes.values()), None, expanded


def rank(item):
    path, token, node, origin = item
    frame = node.frame()
    distance = target_path_distance(frame)
    avatar = avatar_position(frame)
    target_bonus = 0 if distance is None else 1000 - 25 * distance
    expanded = sum(
        _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
        for i in range(10) for j in range(8)
    )
    central = 0 if avatar is None else 4 - abs(33 - avatar[0]) // 6
    screen_row = (
        0 if avatar is None
        else min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - avatar[1]))
    )
    world_row = origin + screen_row
    return (
        target_bonus - 20 * abs(12 - world_row)
        + 3 * expanded + central - len(path)
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    world = {}
    origin = 0
    merge(world, lattice(env.frame()), origin)
    if os.environ.get("ROOT") == "seed":
        root_route = [*SEED, (7,), (7,)]
    elif os.environ.get("ROOT") == "gate":
        root_route = GATE_ROOT
    else:
        root_route = BASE
    for action in root_route:
        env.step(*action)
        origin = alignment(world, lattice(env.frame()), origin)[-1]
        merge(world, lattice(env.frame()), origin)
    base_level = int(env.levels_completed)
    root = env.clone()
    beam = [((), ("base-selected",), root, origin)]
    seen = {frame_key(root.frame(), ("base-selected",), origin)}
    beam_width = int(os.environ.get("BEAM_WIDTH", "6"))
    max_macros = int(os.environ.get("MAX_MACROS", "8"))
    max_local = int(os.environ.get("MAX_LOCAL", "45"))
    generated = 0
    started = time.monotonic()
    for depth in range(1, max_macros + 1):
        candidates = []
        for path, token, node, node_origin in beam:
            successors, winning, expanded = local_successors(
                node, token, node_origin, world, base_level,
                max_local=max_local
            )
            generated += expanded
            if winning is not None:
                route = [*root_route, *path, *winning]
                print("SELECTED_MACRO_WIN", depth, route, flush=True)
                return
            for suffix, child_token, child, child_origin in successors:
                child_path = (*path, *suffix)
                key = frame_key(
                    child.frame(), child_token, child_origin
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    (child_path, child_token, child, child_origin)
                )
        candidates.sort(key=rank, reverse=True)
        beam = candidates[:beam_width]
        print(
            "SELECTED_MACRO_BEAM", depth, generated, len(candidates),
            [
                (
                    rank(item), len(item[0]),
                    avatar_position(item[2].frame()),
                    target_path_distance(item[2].frame()),
                    item[3],
                    item[0],
                )
                for item in beam
            ],
            round(time.monotonic() - started, 1),
            flush=True,
        )
        if not beam:
            break
    print("SELECTED_MACRO_DONE", generated, len(seen), flush=True)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
