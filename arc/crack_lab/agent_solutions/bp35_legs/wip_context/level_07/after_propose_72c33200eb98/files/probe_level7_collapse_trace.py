import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, band_shift, click_action
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


OPENING = [
    click_action(2, 2),
    click_action(4, 2),
    click_action(4, 4),
    click_action(1, 3),
]

TAIL = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
    (3,), (6, 3, 33), (4,), (6, 3, 33),
    (6, 3, 15), (3,), (6, 3, 45),
    (6, 3, 57), (4,), (6, 3, 51),
    (3,), (6, 3, 27), (4,), (6, 3, 57),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(3 + 6 * i - int(ys.mean()))),
        min(range(8), key=lambda j: abs(15 + 6 * j - int(xs.mean()))),
    )


def controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def expanded_supports(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        (i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    node = env.clone()
    for action in OPENING:
        node.step(*action)
    for index in range(len(TAIL) + 1):
        if node.terminal():
            print("BASE_TERMINAL", index)
            break
        before = arr(node.frame()).copy()
        before_avatar = avatar_cell(before)
        before_controls = controls(before)
        for cell in expanded_supports(before):
            child = node.clone()
            child.step(*click_action(*cell))
            if child.terminal():
                print("COLLAPSE", index, cell, "TERMINAL")
                continue
            gain = band_shift(before, child.frame())
            after_avatar = avatar_cell(child.frame())
            after_controls = controls(child.frame())
            if (
                gain
                or after_avatar != before_avatar
                or after_controls != before_controls
                or child.levels_completed > 6
            ):
                print(
                    "COLLAPSE",
                    index,
                    cell,
                    "gain",
                    gain,
                    "avatar",
                    before_avatar,
                    after_avatar,
                    "controls",
                    before_controls,
                    after_controls,
                    "level",
                    child.levels_completed,
                    flush=True,
                )
        if index < len(TAIL):
            node.step(*TAIL[index])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
