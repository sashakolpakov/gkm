import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from probe_level7_no_control import (
    PREFIX, SUFFIX, advance, avatar_cell, controls,
)
from probe_level7_reward_recovery import lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ROUTE40 = [
    *PREFIX,
    (6, 33, 45),
    *SUFFIX,
    (6, 3, 21), (3,), (6, 3, 27),
]

WALKS = [
    [],
    [(3,)], [(3,), (3,)],
    [(4,)], [(4,), (4,)],
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    height = advance(root, ROUTE40)
    ai, aj = avatar_cell(root.frame())
    supports = [
        (i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 2), min(8, aj + 3))
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    print(
        "ROOT", height, avatar_cell(root.frame()), controls(root.frame()),
        [(cell, _cell_shape(root.frame(), *cell)) for cell in supports],
        lattice(root.frame()), flush=True,
    )
    outcomes = []
    for support in [None, *supports]:
        staged = root.clone()
        support_gain = (
            0 if support is None
            else advance(staged, [click_action(*support)])
        )
        if staged.terminal():
            continue
        for walk in WALKS:
            crossed = staged.clone()
            walk_gain = advance(crossed, walk)
            if crossed.terminal():
                continue
            for y in controls(crossed.frame()):
                child = root.clone()
                path = (
                    ([] if support is None else [click_action(*support)])
                    + walk + [(6, 3, y)]
                )
                gain = advance(child, path)
                if child.levels_completed > 6:
                    print("WIN", [*ROUTE40, *path], flush=True)
                    return
                if not child.terminal():
                    outcomes.append(
                        (
                            height + gain, support, walk, y,
                            avatar_cell(child.frame()),
                            tuple(controls(child.frame())),
                            lattice(child.frame()),
                        )
                    )
    outcomes.sort(key=lambda item: (-len(item[5]), -item[0]))
    print("OUTCOMES", len(outcomes))
    for outcome in outcomes:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
