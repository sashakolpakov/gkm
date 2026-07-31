import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import click_action
from perception import arr
from probe_level7_coordinate_decode import advance
from probe_level7_greedy2 import PREFIX, MACRO1
from probe_level7_reward_recovery import avatar_cell, controls, lattice
from probe_level7_stage19_support import KEEP, NEXT


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


BRANCHES = [
    None,
    (4, 4),
    (5, 4),
    (6, 4),
    (7, 4),
]


def simple_successors(root):
    outcomes = {}
    for pre in ([], [(3,)], [(4,)]):
        staged = root.clone()
        pre_gain = advance(staged, pre)
        if staged.terminal():
            continue
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            gain1 = pre_gain + advance(flipped, [(6, 3, y1)])
            path1 = [*pre, (6, 3, y1)]
            if flipped.levels_completed > 6:
                return [], path1
            if flipped.terminal():
                continue
            for cross in ((3,), (4,)):
                middle = flipped.clone()
                gain2 = gain1 + advance(middle, [cross])
                path2 = [*path1, cross]
                if middle.levels_completed > 6:
                    return [], path2
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    child = middle.clone()
                    gain3 = gain2 + advance(child, [(6, 3, y2)])
                    path = [*path2, (6, 3, y2)]
                    if child.levels_completed > 6:
                        return [], path
                    if child.terminal():
                        continue
                    frame = arr(child.frame())
                    key = frame[:63].tobytes()
                    outcomes.setdefault(
                        key,
                        (
                            gain3, len(controls(frame)),
                            avatar_cell(frame), path, child,
                        ),
                    )
    return list(outcomes.values()), None


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root19 = env.clone()
    prefix_route = [*PREFIX, *MACRO1, *KEEP]
    base_height = advance(root19, prefix_route)
    all_outcomes = []
    for support in BRANCHES:
        stage = root19.clone()
        support_path = [] if support is None else [click_action(*support)]
        branch_gain = advance(stage, [*support_path, *NEXT])
        print(
            "BRANCH", support, base_height + branch_gain,
            avatar_cell(stage.frame()), controls(stage.frame()), flush=True,
        )
        options, winning = simple_successors(stage)
        if winning is not None:
            print(
                "WIN", [*prefix_route, *support_path, *NEXT, *winning],
                flush=True,
            )
            return
        for gain, remaining, avatar, suffix, child in options:
            all_outcomes.append(
                (
                    base_height + branch_gain + gain,
                    remaining, support, avatar, suffix,
                    tuple(controls(child.frame())), lattice(child.frame()),
                )
            )
    all_outcomes.sort(key=lambda item: (-item[1], -item[0]))
    print("OUTCOMES", len(all_outcomes))
    for outcome in all_outcomes[:50]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
