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
    base_height = advance(root, [*PREFIX, click_action(5, 2), *SUFFIX])
    ai, aj = avatar_cell(root.frame())
    support_cells = [
        (i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    print(
        "ROOT", base_height, avatar_cell(root.frame()),
        controls(root.frame()),
        [(cell, _cell_shape(root.frame(), *cell)) for cell in support_cells],
        lattice(root.frame()),
    )
    cores = [
        [(3,), (6, 3, 9), (4,), (6, 3, 39)],
        [(4,), (6, 3, 15), (3,), (6, 3, 39)],
    ]
    outcomes = []
    for cell in [None, *support_cells]:
        staged = root.clone()
        support_gain = (
            0 if cell is None else advance(staged, [click_action(*cell)])
        )
        if staged.levels_completed > 6:
            print("WIN_SUPPORT", cell)
            return
        if staged.terminal():
            continue
        for core in cores:
            middle = staged.clone()
            core_gain = support_gain + advance(middle, core)
            if middle.levels_completed > 6:
                print("WIN_CORE", cell, core)
                return
            if middle.terminal():
                continue
            for direction in ((3,), (4,)):
                walked = middle.clone()
                walked_gain = core_gain
                for count in range(6):
                    if walked.levels_completed > 6:
                        print("WIN_WALK", cell, core, direction, count)
                        return
                    if walked.terminal():
                        break
                    visible = controls(walked.frame())
                    for y in visible:
                        child = walked.clone()
                        child.step(6, 3, y)
                        if child.levels_completed > 6:
                            print(
                                "WIN_SWITCH", cell, core, direction,
                                count, y, flush=True,
                            )
                            return
                        if not child.terminal():
                            outcomes.append(
                                (
                                    base_height + walked_gain,
                                    cell, core, direction, count, y,
                                    avatar_cell(child.frame()),
                                    tuple(controls(child.frame())),
                                )
                            )
                    walked_gain += advance(walked, [direction])
    outcomes.sort(
        key=lambda item: (
            -item[0], -len(item[7]),
            -(item[6][1] if item[6] else -1),
        )
    )
    print("NO_CROSS_SUPPORT_WIN", len(outcomes))
    for outcome in outcomes[:30]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
