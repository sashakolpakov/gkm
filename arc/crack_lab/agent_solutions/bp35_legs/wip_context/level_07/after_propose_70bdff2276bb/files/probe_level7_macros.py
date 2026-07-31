import itertools
import json
import os
import sys

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


def gravity_action(frame):
    for y in ROW_ANCHORS:
        if int(frame[y][3]) == 8:
            return CLICK, 3, y
    return None


def nearby_supports(frame):
    ai, aj = avatar_cell(frame)
    out = []
    for i in range(max(0, ai - 2), min(10, ai + 3)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            color, area = _cell_shape(frame, i, j)
            if color in (12, 14) and area < 21:
                out.append(click_action(i, j))
    return out


def execute(root, template):
    node = root.clone()
    route = []
    for token in template:
        action = gravity_action(node.frame()) if token == "g" else token
        if action is None or node.terminal():
            break
        node.step(*action)
        route.append(action)
    return node, route


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    prefix = [
        (4,), (4,), (4,), click_action(8, 4),
        (6, 3, 3), (4,), (6, 3, 3), (4,),
        (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
        (3,),
        (4,), (4,), click_action(8, 2), (6, 3, 9),
        (3,), (3,), (6, 3, 15), (3,), (3,),
    ]
    run_actions(env, prefix)
    start = arr(env.frame()).copy()
    candidates = nearby_supports(start)
    results = []
    for pre_action, pre_count, support, cross_action, cross_count, post_count in (
        itertools.product(((3,), (4,)), range(3), candidates,
                          ((3,), (4,)), range(1, 4), range(3))
    ):
        template = (
            [pre_action] * pre_count
            + [support, "g"]
            + [cross_action] * cross_count
            + ["g"]
            + [cross_action] * post_count
        )
        node, route = execute(env, template)
        shift = band_shift(start, node.frame())
        if node.levels_completed > 6 or (
            not node.terminal() and avatar_cell(node.frame()) is not None and shift
        ):
            results.append((
                node.levels_completed, shift, avatar_cell(node.frame()),
                len(route), route, node.terminal(),
            ))
    results.sort(key=lambda item: (-item[0], -item[1], item[3], item[4]))
    print("START", avatar_cell(start), "SUPPORTS", candidates)
    for result in results[:20]:
        print("MACRO", result)
    print("COUNT", len(results))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
