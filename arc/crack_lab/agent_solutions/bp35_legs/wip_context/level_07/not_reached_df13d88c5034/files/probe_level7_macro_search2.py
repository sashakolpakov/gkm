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


def controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def support_actions(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
    ]


def search(root, max_states=80, max_macros=6):
    def reconstruct(path):
        node = root.clone()
        height = 0
        for action in path:
            if node.terminal():
                break
            before = arr(node.frame()).copy()
            node.step(*action)
            height += band_shift(before, node.frame())
        return node, height

    counter = itertools.count()
    queue = [(0, 0, next(counter), 0, ())]
    seen = set()
    enqueued = set()
    best = (0, ())
    expanded = 0
    while queue and expanded < max_states:
        _, _, _, depth, path = heapq.heappop(queue)
        node, height = reconstruct(path)
        expanded += 1
        if node.levels_completed > 6:
            return list(path), expanded, best
        if node.terminal() or avatar_cell(node.frame()) is None:
            continue
        frame = arr(node.frame())
        print(
            "EXPAND", expanded, height, len(path), avatar_cell(frame),
            controls(frame), flush=True,
        )
        key = height, frame[:63].tobytes(), moves_used(frame) % 2
        if key in seen:
            continue
        seen.add(key)
        if height > best[0]:
            best = height, path
            print("PROGRESS", expanded, height, len(path), path, flush=True)
        if depth >= max_macros:
            continue
        for support in (None, *support_actions(frame)):
            supported = path if support is None else path + (support,)
            for normal_move in (None, 3, 4):
                before_flip = (
                    supported
                    if normal_move is None
                    else supported + ((normal_move,),)
                )
                staged, _ = reconstruct(before_flip)
                if staged.terminal():
                    continue
                for y1 in controls(staged.frame()):
                    for cross_move in (3, 4):
                        prefix = (
                            before_flip + ((6, 3, y1), (cross_move,))
                        )
                        middle, _ = reconstruct(prefix)
                        if middle.terminal():
                            continue
                        for y2 in controls(middle.frame()):
                            child_path = prefix + ((6, 3, y2),)
                            child, child_height = reconstruct(child_path)
                            avatar = avatar_cell(child.frame())
                            if child.levels_completed > 6:
                                return list(child_path), expanded, best
                            if (
                                child.terminal() or avatar is None
                                or avatar[0] != 6
                            ):
                                continue
                            child_frame = arr(child.frame())
                            child_key = (
                                child_height, child_frame[:63].tobytes(),
                                moves_used(child_frame) % 2,
                            )
                            if child_key in enqueued:
                                continue
                            enqueued.add(child_key)
                            heapq.heappush(
                                queue,
                                (
                                    -child_height, len(child_path),
                                    next(counter), depth + 1, child_path,
                                ),
                            )
    return [], expanded, best


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(
        env, [click_action(2, 2), *PREFIX, (6, 3, 27)]
    )
    route, expanded, best = search(env)
    verified = env.clone()
    run_actions(verified, route)
    print("SEARCH", expanded, len(route), route)
    print(
        "BEST", best[0], len(best[1]), best[1],
        "VERIFY", verified.levels_completed, verified.terminal(),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
