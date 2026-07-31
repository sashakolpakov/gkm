import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
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

    root = env.clone()
    base_height = advance(root, PREFIX)
    print(
        "ROOT", base_height, avatar_cell(root.frame()), controls(root.frame()),
        (5, 2, _cell_shape(root.frame(), 5, 2)),
        (5, 4, _cell_shape(root.frame(), 5, 4)),
        lattice(root.frame()),
    )
    stages = [
        [click_action(5, 2), click_action(5, 4)],
        [click_action(5, 4), click_action(5, 2)],
    ]
    cores = [
        [(3,), (6, 3, 9), (4,), (6, 3, 39)],
        [(4,), (6, 3, 15), (3,), (6, 3, 39)],
    ]
    outcomes = []
    for stage in stages:
        staged = root.clone()
        stage_gain = advance(staged, [*stage, *SUFFIX])
        print(
            "STAGED", stage, base_height + stage_gain,
            staged.levels_completed, staged.terminal(),
            None if staged.terminal() else avatar_cell(staged.frame()),
            [] if staged.terminal() else controls(staged.frame()),
            "" if staged.terminal() else lattice(staged.frame()),
        )
        if staged.levels_completed > 6:
            print("WIN_STAGE", stage)
            return
        if staged.terminal():
            continue
        for core in cores:
            middle = staged.clone()
            core_gain = stage_gain + advance(middle, core)
            if middle.levels_completed > 6:
                print("WIN_CORE", stage, core)
                return
            if middle.terminal():
                continue
            for direction in ((3,), (4,)):
                walked = middle.clone()
                walked_gain = core_gain
                for count in range(7):
                    if walked.levels_completed > 6:
                        print("WIN_WALK", stage, core, direction, count)
                        return
                    if walked.terminal():
                        break
                    for y in controls(walked.frame()):
                        child = walked.clone()
                        child.step(6, 3, y)
                        outcome = (
                            base_height + walked_gain,
                            stage, core, direction, count, y,
                            child.levels_completed,
                            not child.terminal(),
                            None if child.terminal()
                            else avatar_cell(child.frame()),
                            () if child.terminal()
                            else tuple(controls(child.frame())),
                        )
                        outcomes.append(outcome)
                        if child.levels_completed > 6:
                            print(
                                "WIN_SWITCH", stage, core, direction,
                                count, y, flush=True,
                            )
                            return
                    walked_gain += advance(walked, [direction])
    outcomes.sort(key=lambda item: (-item[6], -item[7], -item[0]))
    print("NO_DUAL_STAGE_WIN", len(outcomes))
    for outcome in outcomes[:30]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
