import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from probe_level7_no_control import (
    PREFIX, SUFFIX, advance, avatar_cell, controls,
)
from probe_level7_reward_recovery import lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


STAGE = (6, 33, 45)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    base_height = advance(root, [*PREFIX, STAGE, *SUFFIX])
    outcomes = []
    for pre in ([], [(3,)], [(4,)]):
        pre_node = root.clone()
        advance(pre_node, pre)
        for y1 in controls(pre_node.frame()):
            for cross in ([], [(3,)], [(4,)]):
                middle = root.clone()
                gain = advance(
                    middle, [*pre, (6, 3, y1), *cross]
                )
                if middle.levels_completed > 6:
                    print("WIN_FIRST", pre, y1, cross)
                    return
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    child = root.clone()
                    path = [
                        *pre, (6, 3, y1), *cross, (6, 3, y2)
                    ]
                    total = advance(child, path)
                    if child.levels_completed > 6:
                        print(
                            "WIN_SECOND",
                            [*PREFIX, STAGE, *SUFFIX, *path],
                            flush=True,
                        )
                        return
                    if child.terminal():
                        continue
                    outcomes.append(
                        (
                            base_height + total,
                            avatar_cell(child.frame()),
                            tuple(controls(child.frame())),
                            path,
                            lattice(child.frame()),
                        )
                    )
    outcomes.sort(key=lambda item: (-item[0], -len(item[2])))
    print("OUTCOMES", len(outcomes))
    for outcome in outcomes:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
