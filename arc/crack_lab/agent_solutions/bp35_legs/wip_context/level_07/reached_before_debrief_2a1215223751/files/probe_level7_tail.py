import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
    run_actions,
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
]
CYCLE = [(3,), "gravity", (4,), "gravity"]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def gravity_action(frame):
    return next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )


def run_dynamic(node, route):
    shift = 0
    for token in route:
        action = gravity_action(node.frame()) if token == "gravity" else token
        if action is None or node.terminal():
            return shift
        before = arr(node.frame()).copy()
        node.step(*action)
        shift += (
            band_shift(before, node.frame())
            - band_shift(node.frame(), before)
        )
    return shift


def summary(node):
    frame = node.frame()
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
        12: "c", 14: "Y", 15: "f",
    }
    grid = "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )
    shaped = [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14, 15)
    ]
    return node.levels_completed, node.terminal(), avatar_cell(frame), grid, shaped


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print("START", summary(env))
    print("ACTIONS", env.actions)
    for action in (3, 4):
        node = env.clone()
        before = arr(node.frame()).copy()
        node.step(action)
        print("KEY", action, band_shift(before, node.frame()), summary(node)[:3])

    start = env.clone()
    node = start.clone()
    prefix_shift = run_dynamic(node, PREFIX)
    print(
        "PREFIX", prefix_shift, gravity_action(node.frame()), summary(node)
    )
    before_cycles = node.clone()
    for index in range(2):
        shift = 0
        for token in CYCLE:
            action = (
                gravity_action(node.frame())
                if token == "gravity" else token
            )
            if action is None or node.terminal():
                break
            before = arr(node.frame()).copy()
            node.step(*action)
            delta = (
                band_shift(before, node.frame())
                - band_shift(node.frame(), before)
            )
            shift += delta
            print(
                "TRACE", index + 1, action, delta,
                gravity_action(node.frame()), summary(node)[:4],
            )
        print(
            "CYCLE", index + 1, shift,
            gravity_action(node.frame()), summary(node),
        )
    tail = node
    for route in (
        [(3,)], [(3,), (3,)], [(3,), (4,)],
        [(4,)], [(4,), (3,)], [(4,), (4,)],
    ):
        moved = tail.clone()
        shift = run_dynamic(moved, route)
        print("MOVE", route, shift, summary(moved)[:4])
    for cell in ((4, 1), (5, 1), (5, 2), (5, 4), (6, 1), (6, 3)):
        for suffix in ((), ((3,),), ((4,),)):
            moved = tail.clone()
            route = [click_action(*cell), *suffix]
            shift = run_dynamic(moved, route)
            print("SUPPORT", cell, suffix, shift, summary(moved)[:4])
    inverted = before_cycles.clone()
    run_dynamic(inverted, [(3,), "gravity", (4,)])
    for cell in ((2, 2), (2, 4), (3, 3), (8, 2), (8, 3)):
        for rights in range(3):
            moved = inverted.clone()
            route = [click_action(*cell), *([(4,)] * rights), "gravity"]
            shift = run_dynamic(moved, route)
            print("INVERTED", cell, rights, shift, summary(moved)[:4])
    support_cells = [
        (i, j) for i in range(10) for j in range(8)
        if _cell_shape(before_cycles.frame(), i, j)[0] in (12, 14)
    ]
    for cell in support_cells:
        moved = before_cycles.clone()
        run_dynamic(moved, [click_action(*cell)])
        shift = 0
        for _ in range(2):
            shift += run_dynamic(moved, CYCLE)
        print(
            "STAGE", cell, shift, gravity_action(moved.frame()),
            summary(moved)[:4],
        )
    for staged in (
        ((6, 4),),
        ((0, 2),),
        ((0, 3),),
        ((0, 4),),
        ((6, 4), (0, 2)),
        ((6, 4), (0, 3)),
        ((6, 4), (0, 4)),
    ):
        moved = start.clone()
        run_dynamic(moved, [click_action(*cell) for cell in staged])
        shift = run_dynamic(moved, PREFIX)
        for _ in range(2):
            shift += run_dynamic(moved, CYCLE)
        print(
            "START_STAGE", staged, shift, gravity_action(moved.frame()),
            summary(moved)[:4],
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
