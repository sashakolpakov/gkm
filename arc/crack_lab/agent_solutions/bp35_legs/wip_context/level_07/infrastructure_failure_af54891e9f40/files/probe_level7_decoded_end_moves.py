import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    height = advance(root, decoded_route())
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
        lattice(root.frame()),
    )
    outcomes = {}
    for support in [None, *supports]:
        staged = root.clone()
        support_gain = (
            0 if support is None
            else advance(staged, [click_action(*support)])
        )
        if staged.levels_completed > 6:
            print("WIN_SUPPORT", support)
            return
        if staged.terminal():
            continue
        for direction in ((3,), (4,)):
            walked = staged.clone()
            walked_gain = support_gain
            for count in range(7):
                if walked.levels_completed > 6:
                    print("WIN_WALK", support, direction, count)
                    return
                if walked.terminal():
                    break
                key = (
                    height + walked_gain,
                    avatar_cell(walked.frame()),
                    tuple(controls(walked.frame())),
                    lattice(walked.frame()),
                )
                outcomes.setdefault(key, (support, direction, count))
                walked_gain += advance(walked, [direction])
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (-item[0][0], -len(item[0][2])),
    )
    print("END_OUTCOMES", len(ordered))
    for state, source in ordered:
        print("OPTION", state[:3], source, state[3])

    for rights in (0, 1):
        staged = root.clone()
        advance(staged, [(4,)] * rights)
        for y in controls(staged.frame()):
            child = staged.clone()
            gain = advance(child, [(6, 3, y)])
            print(
                "FINAL_FLIP", rights, y, height + gain,
                child.levels_completed, child.terminal(),
                None if child.terminal() else avatar_cell(child.frame()),
                [] if child.terminal() else controls(child.frame()),
                "" if child.terminal() else lattice(child.frame()),
            )
            if child.levels_completed > 6:
                print(
                    "WIN", [*decoded_route(), *([(4,)] * rights),
                            (6, 3, y)],
                    flush=True,
                )
                return
            if rights == 1 and not child.terminal():
                for direction in ((3,), (4,)):
                    walked = child.clone()
                    walked_gain = 0
                    for count in range(5):
                        print(
                            "HIGH_WALK", y, direction, count,
                            height + gain + walked_gain,
                            walked.levels_completed, walked.terminal(),
                            None if walked.terminal()
                            else avatar_cell(walked.frame()),
                            [] if walked.terminal()
                            else controls(walked.frame()),
                        )
                        if (
                            walked.levels_completed > 6
                            or walked.terminal()
                        ):
                            break
                        walked_gain += advance(walked, [direction])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
