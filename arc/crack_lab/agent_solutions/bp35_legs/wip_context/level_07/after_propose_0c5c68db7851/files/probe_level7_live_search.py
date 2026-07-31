import heapq
import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    ROW_ANCHORS, _cell_shape, band_shift, click_action, moves_used,
    run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    click_action(2, 2),
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
        min(range(8), key=lambda j: abs(15 + 6 * j - int(xs.mean()))),
    )


def controls(frame):
    return tuple(y for y in ROW_ANCHORS if int(frame[y][3]) == 8)


def actions(frame):
    out = [(3,), (4,)]
    visible_controls = controls(frame)
    out.extend((6, 3, y) for y in visible_controls)
    if not visible_controls:
        return out
    avatar = avatar_cell(frame)
    if avatar is None:
        return out
    ai, aj = avatar
    out.extend(
        click_action(i, j)
        for i in range(max(0, ai - 1), min(10, ai + 2))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    )
    return out


def state_key(frame, height):
    return (
        height,
        arr(frame)[:63].tobytes(),
        moves_used(frame) % 2,
    )


def priority(height, frame, depth):
    avatar = avatar_cell(frame)
    remaining = len(controls(frame))
    central = 0 if avatar is None else min(avatar[1], 7 - avatar[1], 3)
    score = height * 10 + remaining * 20 + central
    return -score, depth


def search(root, max_expansions=600, max_depth=58):
    counter = itertools.count()
    start_frame = arr(root.frame()).copy()
    queue = [(*priority(0, start_frame, 0), next(counter), root.clone(), (), 0)]
    seen = {state_key(start_frame, 0)}
    best = (0, (), avatar_cell(start_frame), controls(start_frame))
    expanded = 0
    while queue and expanded < max_expansions:
        _, _, _, node, path, height = heapq.heappop(queue)
        if len(path) >= max_depth:
            continue
        before = arr(node.frame()).copy()
        for action in actions(before):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > 6:
                return child_path, best, expanded
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            child_frame = arr(child.frame()).copy()
            child_height = height + band_shift(before, child_frame)
            key = state_key(child_frame, child_height)
            if key in seen:
                continue
            seen.add(key)
            if (
                child_height,
                len(controls(child_frame)),
                -len(child_path),
            ) > (
                best[0],
                len(best[3]),
                -len(best[1]),
            ):
                best = (
                    child_height,
                    child_path,
                    avatar_cell(child_frame),
                    controls(child_frame),
                )
                print("PROGRESS", expanded, best, flush=True)
            heapq.heappush(
                queue,
                (
                    *priority(child_height, child_frame, len(child_path)),
                    next(counter), child, child_path, child_height,
                ),
            )
            if expanded >= max_expansions:
                break
    return (), best, expanded


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    run_actions(root, PREFIX)
    print(
        "ROOT", avatar_cell(root.frame()), controls(root.frame()),
        len(PREFIX), flush=True,
    )
    route, best, expanded = search(root)
    print("SEARCH", expanded, len(route), route)
    print("BEST", best)
    if route:
        verified = root.clone()
        run_actions(verified, route)
        print(
            "VERIFY", verified.levels_completed, verified.terminal(),
            avatar_cell(verified.frame()), controls(verified.frame()),
        )
        print("WIN", [*PREFIX, *route], flush=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
