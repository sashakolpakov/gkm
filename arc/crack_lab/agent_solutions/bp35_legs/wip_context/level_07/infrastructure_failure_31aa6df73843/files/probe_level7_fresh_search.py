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
    moves_used,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)["final_path"]

PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def evaluate(suffix):
    observation = {}

    def replay(env):
        for action in CHECKPOINT:
            env.step(action)
        for action in PREFIX:
            env.step(*action)
        height = 0
        for action in suffix:
            if env.terminal():
                break
            before = arr(env.frame()).copy()
            env.step(*action)
            height += band_shift(before, env.frame())
        frame = arr(env.frame()).copy()
        observation.update(
            frame=frame,
            height=height,
            level=env.levels_completed,
            terminal=env.terminal(),
            avatar=avatar_cell(frame),
        )

    _, _, err = A.run_program("bp35", replay)
    observation["err"] = err
    return observation


def choices(frame):
    out = [(3,), (4,)]
    out.extend(
        (6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8
    )
    avatar = avatar_cell(frame)
    if avatar is None:
        return out
    ai, aj = avatar
    for i in range(max(0, ai - 2), min(10, ai + 3)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            if _cell_shape(frame, i, j)[0] in (12, 14):
                out.append(click_action(i, j))
    return out


def search(max_states=120, max_depth=32):
    counter = itertools.count()
    queue = [(0, 0, next(counter), ())]
    seen = set()
    best = (-1, ())
    for expanded in range(max_states):
        if not queue:
            break
        _, _, _, path = heapq.heappop(queue)
        result = evaluate(path)
        if result["err"] is not None:
            continue
        if result["level"] > 6:
            return list(path), expanded + 1, best
        if result["terminal"] or result["avatar"] is None:
            continue
        frame = result["frame"]
        key = (
            result["height"], frame[:63].tobytes(), moves_used(frame) % 2,
        )
        if key in seen:
            continue
        seen.add(key)
        if result["height"] > best[0]:
            best = (result["height"], path)
            print(
                "PROGRESS", expanded + 1, result["height"], len(path), path,
                flush=True,
            )
        if len(path) >= max_depth:
            continue
        for action in choices(frame):
            child_path = path + (action,)
            heapq.heappush(
                queue,
                (-result["height"], len(child_path), next(counter), child_path),
            )
    return [], max_states, best


route, expanded, best = search()
verified = evaluate(route)
print("SEARCH", expanded, len(route), route)
print(
    "BEST", best[0], len(best[1]), best[1],
    "VERIFY", verified.get("level"), verified.get("terminal"),
)
