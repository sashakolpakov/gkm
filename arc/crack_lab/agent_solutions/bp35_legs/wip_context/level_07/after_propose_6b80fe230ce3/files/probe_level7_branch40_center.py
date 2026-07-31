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


ROOT_SUFFIXES = [
    [(3,), (6, 3, 45), (4,), (6, 3, 27)],
    [(3,), (6, 3, 45), (4,), (6, 3, 51)],
]
WALKS = [[], [(3,)], [(4,)], [(3,), (3,)], [(4,), (4,)]]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    outcomes = []
    for root_suffix in ROOT_SUFFIXES:
        root = env.clone()
        route = [*PREFIX, (6, 33, 45), *SUFFIX, *root_suffix]
        height = advance(root, route)
        ai, aj = avatar_cell(root.frame())
        supports = [
            (i, j)
            for i in range(max(0, ai - 2), min(10, ai + 3))
            for j in range(max(0, aj - 2), min(8, aj + 3))
            if _cell_shape(root.frame(), i, j)[0] in (12, 14)
        ]
        print(
            "ROOT", root_suffix, height, avatar_cell(root.frame()),
            controls(root.frame()),
            [(cell, _cell_shape(root.frame(), *cell))
             for cell in supports],
            lattice(root.frame()), flush=True,
        )
        for support in [None, *supports]:
            support_path = (
                [] if support is None else [click_action(*support)]
            )
            for walk in WALKS:
                staged = root.clone()
                gain = advance(staged, [*support_path, *walk])
                if staged.terminal():
                    continue
                for y in controls(staged.frame()):
                    child = root.clone()
                    path = [*support_path, *walk, (6, 3, y)]
                    total = advance(child, path)
                    if child.levels_completed > 6:
                        print("WIN", [*route, *path], flush=True)
                        return
                    if not child.terminal():
                        outcomes.append(
                            (
                                height + total, root_suffix, support,
                                walk, y, avatar_cell(child.frame()),
                                tuple(controls(child.frame())),
                                lattice(child.frame()),
                            )
                        )
    outcomes.sort(key=lambda item: (-len(item[6]), -item[0]))
    print("OUTCOMES", len(outcomes))
    for outcome in outcomes[:60]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
