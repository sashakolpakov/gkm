import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from probe_level7_reward_recovery import (
    PREFIX, SUFFIX, advance, avatar_cell, controls, lattice,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    node = env.clone()
    route = [
        *PREFIX,
        click_action(5, 2),
        *SUFFIX,
        (3,), (6, 3, 9), (4,), (6, 3, 39),
        (3,), (3,), (3,),
        (6, 3, 27),
    ]
    advance(node, route)
    print(
        "SIDE_ROOM", len(route), node.levels_completed, node.terminal(),
        avatar_cell(node.frame()), controls(node.frame()), lattice(node.frame()),
    )
    barriers = [
        (i, j)
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(node.frame()[y][x]) == 0
    ]
    for cell in barriers:
        clicked = node.clone()
        before = _cell_shape(clicked.frame(), *cell)
        clicked.step(*click_action(*cell))
        after = (
            None if clicked.terminal()
            else _cell_shape(clicked.frame(), *cell)
        )
        print(
            "CLICK", cell, before, after, clicked.levels_completed,
            clicked.terminal(),
            None if clicked.terminal() else avatar_cell(clicked.frame()),
            [] if clicked.terminal() else controls(clicked.frame()),
            "" if clicked.terminal() else lattice(clicked.frame()),
        )
        if clicked.terminal():
            continue
        for direction in ((3,), (4,)):
            child = clicked.clone()
            advance(child, [direction])
            print(
                "MOVE", cell, direction, child.levels_completed,
                child.terminal(),
                None if child.terminal() else avatar_cell(child.frame()),
                [] if child.terminal() else controls(child.frame()),
            )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
