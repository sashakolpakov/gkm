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


TAIL = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
]

PREFIX = [
    click_action(2, 2),
    click_action(4, 2),
    click_action(4, 4),
    click_action(1, 3),
    *TAIL,
    (6, 39, 27),
    (3,),
    (6, 33, 39),
    click_action(5, 2),
    (6, 33, 57), (6, 3, 33), (4,), (6, 3, 45),
    (4,), (4,),
    (3,), (6, 3, 9), (3,), (6, 3, 51),
    (3,), (6, 3, 15), (4,), (6, 3, 51),
    (3,), (6, 3, 21), (4,), (6, 3, 57),
]

BRANCH = [
    (6, 27, 33),
    (4,), (6, 3, 57), (3,), (6, 3, 15),
    (3,), (6, 3, 27), (4,), (6, 3, 39),
    (3,), (6, 3, 9), (4,), (6, 3, 39),
    (3,), (3,), (3,),
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
    return tuple(y for y in ROW_ANCHORS if int(frame[y][3]) == 8)


def supports(frame):
    return [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
    ]


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 8: "g", 9: "A", 10: ".",
        11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, [*PREFIX, *BRANCH])
    base = arr(env.frame()).copy()
    print(
        "ROOT", env.levels_completed, env.terminal(), avatar_cell(base),
        controls(base), supports(base), lattice(base),
    )
    for direction in ((3,), (4,)):
        child = env.clone()
        for count in range(1, 7):
            before = arr(child.frame()).copy()
            child.step(*direction)
            print(
                "MOVE", direction, count, child.levels_completed,
                not child.terminal(),
                0 if child.terminal() else band_shift(before, child.frame()),
                None if child.terminal() else avatar_cell(child.frame()),
                () if child.terminal() else controls(child.frame()),
            )
            if child.terminal() or child.levels_completed > 6:
                break

    for i, j, shape in supports(base):
        if shape[1] >= 21:
            continue
        child = env.clone()
        child.step(*click_action(i, j))
        before = arr(child.frame()).copy()
        child.step(4)
        print(
            "SUPPORT_R", (i, j), shape,
            child.levels_completed, not child.terminal(),
            0 if child.terminal() else band_shift(before, child.frame()),
            None if child.terminal() else avatar_cell(child.frame()),
            () if child.terminal() else controls(child.frame()),
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
