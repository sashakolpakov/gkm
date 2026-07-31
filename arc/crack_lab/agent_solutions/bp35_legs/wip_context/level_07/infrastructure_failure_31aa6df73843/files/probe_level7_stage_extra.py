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

PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,),
]

END = [
    click_action(5, 2),
    (6, 3, 33), (4,), (6, 3, 45),
    (6, 33, 33),
    (6, 33, 33),
    (6, 3, 21), (4,), (6, 3, 27),
    (6, 3, 27), (4,), (6, 3, 33),
    (6, 3, 51), (4,), (6, 3, 39),
    (6, 3, 15), (4,), (6, 3, 51),
    (6, 3, 9), (4,), (6, 3, 57),
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


def thin_supports(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def expanded_supports(frame):
    return [
        (i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    ]


def execute(root, route):
    node = root.clone()
    height = 0
    for action in route:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            height += band_shift(before, node.frame())
    if node.terminal():
        return node, (node.levels_completed, False, height, 0, 0)
    frame = node.frame()
    expanded = sum(
        1
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    )
    return node, (
        node.levels_completed,
        True,
        height,
        len(controls(frame)),
        expanded,
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_route = [*OPENING, *PREFIX, *END]
    _, baseline = execute(env, base_route)
    print("BASE", baseline, flush=True)

    candidates = []
    for index in range(len(PREFIX) + 1):
        stage, _ = execute(env, [*OPENING, *PREFIX[:index]])
        if stage.terminal():
            continue
        for support in thin_supports(stage.frame()):
            route = [
                *OPENING,
                *PREFIX[:index],
                support,
                *PREFIX[index:],
                *END,
            ]
            node, result = execute(env, route)
            if node.levels_completed > 6:
                print("WIN", index, support, route, flush=True)
                return
            candidates.append((*result, index, support))
    candidates.sort(
        key=lambda item: (
            -item[0], -item[1], -item[2], -item[3], -item[4],
            item[5], item[6],
        )
    )
    print("CANDIDATES", len(candidates))
    for candidate in candidates[:30]:
        print("EXTRA", candidate)

    chosen_index = 15
    chosen_support = (6, 33, 39)
    chosen_route = [
        *OPENING,
        *PREFIX[:chosen_index],
        chosen_support,
        *PREFIX[chosen_index:],
        *END,
    ]
    final, _ = execute(env, chosen_route)
    final_frame = arr(final.frame()).copy()
    surviving = expanded_supports(final_frame)
    print("FINAL_EXPANDED", surviving)
    for cell in surviving:
        child = final.clone()
        child.step(*click_action(*cell))
        if child.terminal():
            print("REMOTE", cell, "TERMINAL")
            continue
        print(
            "REMOTE",
            cell,
            child.levels_completed,
            band_shift(final_frame, child.frame()),
            avatar_cell(child.frame()),
            controls(child.frame()),
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
