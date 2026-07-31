import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    click_action(2, 2),
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
]

MACRO1 = [(3,), (6, 3, 33), (4,), (6, 3, 33)]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    height = advance(root, [*PREFIX, *MACRO1])
    print(
        "ROOT", height, avatar_cell(root.frame()), controls(root.frame()),
        lattice(root.frame()), flush=True,
    )
    outcomes = []
    for pre in ([], [(3,)], [(4,)]):
        staged = root.clone()
        pre_gain = advance(staged, pre)
        if staged.terminal():
            continue
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            gain1 = pre_gain + advance(flipped, [(6, 3, y1)])
            if flipped.levels_completed > 6:
                print("WIN_FIRST", pre, y1)
                return
            if flipped.terminal():
                continue
            for cross in ((3,), (4,)):
                middle = flipped.clone()
                gain2 = gain1 + advance(middle, [cross])
                if middle.levels_completed > 6:
                    print("WIN_CROSS", pre, y1, cross)
                    return
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    child = middle.clone()
                    gain3 = gain2 + advance(child, [(6, 3, y2)])
                    path = [*pre, (6, 3, y1), cross, (6, 3, y2)]
                    if child.levels_completed > 6:
                        print("WIN_SECOND", path)
                        return
                    if child.terminal():
                        continue
                    outcomes.append(
                        (
                            height + gain3,
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


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
