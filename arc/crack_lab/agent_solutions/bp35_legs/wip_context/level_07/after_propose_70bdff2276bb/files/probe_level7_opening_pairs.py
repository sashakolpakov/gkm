import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, band_shift, click_action
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


SAFE = [
    (0, 2), (0, 3), (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4), (3, 3), (4, 2), (4, 4), (6, 4), (8, 4),
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


def execute(root, pair):
    node = root.clone()
    height = 0
    actions = [
        click_action(2, 2),
        *(click_action(*cell) for cell in pair),
        *TAIL,
    ]
    for action in actions:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            height += band_shift(before, node.frame())
    if node.terminal():
        return node.levels_completed, False, height, None, []
    return (
        node.levels_completed,
        True,
        height,
        avatar_cell(node.frame()),
        controls(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    fixed = ((4, 2), (4, 4), (1, 3))
    outcomes = [
        (*execute(env, (*fixed, cell)), (*fixed, cell))
        for cell in SAFE
        if cell not in fixed
    ]
    outcomes.sort(
        key=lambda item: (
            -item[0], -item[1], -len(item[4]), -item[2], item[5],
        )
    )
    for outcome in outcomes[:30]:
        print("PAIR", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
