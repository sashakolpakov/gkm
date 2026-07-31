"""Choose a safe opening through the full-width color-12 barrier."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr
from probe_l8_cross2 import LEFT_EXIT
from probe_l8_stage1 import lattice, target
from probe_l8_stage4 import PREFIX


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ROOT_ROUTE = [*PREFIX, *LEFT_EXIT]


def walk_actions(start, end):
    action = (4,) if end > start else (3,)
    return [action] * abs(end - start)


def run(node, route, trace=False):
    height = 0
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        shape = None
        if len(action) == 3 and action[1] >= 15:
            shape = _cell_shape(
                before, (action[2] - 3) // 6, (action[1] - 15) // 6
            )
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        height += shift
        if trace:
            print(
                "STEP",
                index,
                action,
                "shape",
                shape,
                "alive",
                not node.terminal(),
                "height",
                height,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    outcomes = []
    for column in range(8):
        suffix = [*walk_actions(4, column), click_action(5, column)]
        node = env.clone()
        gain = run(node, [*ROOT_ROUTE, *suffix])
        outcome = (
            column,
            not node.terminal(),
            node.levels_completed,
            gain,
            None if node.terminal() else avatar_column(node.frame()),
            None if node.terminal() else target(node.frame()),
            "" if node.terminal() else lattice(node.frame()),
            suffix,
        )
        outcomes.append(outcome)
        print("CASE", outcome)

    alive = [item for item in outcomes if item[1]]
    best = max(alive, key=lambda item: (item[3], -len(item[-1])))
    print("BEST", best)
    print("DIRECT")
    run(env, [*ROOT_ROUTE, *best[-1]], trace=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
