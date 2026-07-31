import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action,
)
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

    route = [
        *PREFIX,
        click_action(5, 2),
        *SUFFIX,
        (3,), (6, 3, 9), (4,), (6, 3, 39),
        (4,), (4,),
    ]
    root = env.clone()
    height = advance(root, route)
    print(
        "ROOT", len(route), height, root.levels_completed, root.terminal(),
        avatar_cell(root.frame()), controls(root.frame()), lattice(root.frame()),
    )
    frame = root.frame()
    click_candidates = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (0, 12, 14, 15)
    ]
    walk_patterns = [
        [],
        [(3,)], [(3,), (3,)], [(3,), (3,), (3,)],
        [(4,)], [(4,), (4,)], [(4,), (4,), (4,)],
        [(3,), (4,)], [(4,), (3,)],
    ]
    outcomes = {}
    for click in [None, *click_candidates]:
        staged = root.clone()
        click_gain = 0 if click is None else advance(staged, [click])
        if staged.terminal():
            continue
        for walk in walk_patterns:
            child = staged.clone()
            gain = click_gain + advance(child, walk)
            path = ([] if click is None else [click]) + walk
            if child.levels_completed > 6:
                print("WIN", path, flush=True)
                return
            if child.terminal():
                continue
            key = (
                gain,
                avatar_cell(child.frame()),
                tuple(controls(child.frame())),
                lattice(child.frame()),
            )
            outcomes.setdefault(key, path)
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (
            -item[0][0], -len(item[0][2]),
            -(item[0][1][1] if item[0][1] else -1),
            len(item[1]),
        ),
    )
    print("OUTCOMES", len(ordered))
    for state, path in ordered[:30]:
        print(
            "OPTION", height + state[0], state[1], state[2], path, state[3]
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
