import heapq
import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
    moves_used, run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


GRAVITY = (6, 3, 3)
PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), GRAVITY, (4,), GRAVITY, (4,),
    (3,), (3,), click_action(8, 3), GRAVITY, (3,), (6, 3, 9), (3,),
    (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (3,),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def actions(frame):
    out = [(3,), (4,)]
    avatar = avatar_cell(frame)
    if avatar is None:
        return out
    ai, aj = avatar
    for i in range(max(0, ai - 2), min(10, ai + 3)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            color, area = _cell_shape(frame, i, j)
            if color in (12, 14):
                out.append(click_action(i, j))
    gravity = next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )
    if gravity is not None:
        out.append(gravity)
    return out


def search(root, max_expansions=300, max_depth=48):
    counter = itertools.count()
    start = root.clone()
    queue = [(0, 0, next(counter), 0, (), start)]
    frame = arr(start.frame())
    seen = {(0, frame[:63].tobytes(), moves_used(frame) % 2)}
    best = (0, ())
    for expanded in range(max_expansions):
        if not queue:
            return [], expanded, best
        _, _, _, height, path, node = heapq.heappop(queue)
        if len(path) >= max_depth:
            continue
        for action in actions(node.frame()):
            child = node.clone()
            before = arr(node.frame()).copy()
            child.step(*action)
            child_path = path + (action,)
            if child.levels_completed > 6:
                return list(child_path), expanded + 1, (height, child_path)
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            gained = height + band_shift(before, child.frame())
            if gained > best[0]:
                best = (gained, child_path)
                print(
                    "PROGRESS", expanded + 1, gained, len(child_path),
                    child_path,
                )
            frame = arr(child.frame())
            key = (gained, frame[:63].tobytes(), moves_used(frame) % 2)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(
                queue,
                (-gained, len(child_path), next(counter), gained, child_path, child),
            )
    return [], max_expansions, best


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, PREFIX)
    before_shape = _cell_shape(env.frame(), 7, 2)
    toggled = env.clone()
    toggled.step(*click_action(7, 2))
    print("TOGGLE", before_shape, _cell_shape(toggled.frame(), 7, 2))
    route, expanded, best = search(env)
    verified = env.clone()
    run_actions(verified, route)
    print("SEARCH", expanded, len(route), route)
    print(
        "BEST", best[0], len(best[1]), "VERIFY",
        verified.levels_completed, verified.terminal(),
        avatar_cell(verified.frame()),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
