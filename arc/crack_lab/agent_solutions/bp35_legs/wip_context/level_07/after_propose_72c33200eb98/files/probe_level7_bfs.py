import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    CLICK, COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift,
    click_action, run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def actions(node):
    frame = node.frame()
    out = [(3,), (4,)]
    avatar = avatar_cell(frame)
    if avatar is None:
        return out
    ai, aj = avatar
    for i in range(max(0, ai - 2), min(10, ai + 3)):
        y = ROW_ANCHORS[i]
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            x = COL_ANCHORS[j]
            color, area = _cell_shape(frame, i, j)
            if (
                color in (12, 14)
                and area < 21
                and abs(i - ai) <= 2
                and abs(j - aj) <= 1
            ):
                out.append((CLICK, x, y))
            elif (
                color == 15
                and abs(i - ai) <= 1
                and abs(j - aj) <= 1
            ):
                out.append((CLICK, x, y))
    for y in ROW_ANCHORS:
        if int(frame[y][3]) == 8:
            out.append((CLICK, 3, y))
            break
    return out


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    opener = [
        (4,), (4,), (4,), click_action(8, 4),
        (6, 3, 3), (4,), (6, 3, 3), (4,),
        (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
        (3,),
        (4,), (4,), click_action(8, 2), (6, 3, 9),
        (3,), (3,), (6, 3, 15), (3,), (3,),
        (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
        (3,),
    ]
    run_actions(env, opener)
    start = arr(env.frame()).copy()

    def goal(node, path):
        if node.levels_completed > 6:
            return True
        avatar = avatar_cell(node.frame())
        return (
            not node.terminal()
            and avatar is not None
            and avatar[0] == 6
            and band_shift(start, node.frame()) > 0
        )

    def reconstruct(path):
        node = env.clone()
        run_actions(node, path)
        return node

    queue = deque([()])
    seen = {start[:63].tobytes()}
    route = None
    expanded = 0
    while queue and expanded < 300:
        path = queue.popleft()
        if len(path) >= 10:
            continue
        for action in actions(reconstruct(path)):
            child_path = path + (action,)
            child = reconstruct(child_path)
            expanded += 1
            if goal(child, child_path):
                route = list(child_path)
                queue.clear()
                break
            if child.terminal():
                continue
            key = arr(child.frame())[:63].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append(child_path)
        if route is not None:
            break
    verified = env.clone()
    run_actions(verified, route or [])
    print("START", avatar_cell(env.frame()), "EXPANDED", expanded)
    print("ROUTE", route, "ACTIONS", actions(env))
    print(
        "VERIFY",
        band_shift(start, verified.frame()),
        avatar_cell(verified.frame()),
        verified.levels_completed,
        verified.terminal(),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
