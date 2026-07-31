import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift, click_action
from perception import arr
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

    base = decoded_route()
    route = [
        *base[:40], click_action(6, 2), *base[40:],
        (3,), (3,), (3,), (3,),
    ]
    root = env.clone()
    height = advance(root, route)
    print(
        "ROOT", len(route), height, root.levels_completed, root.terminal(),
        avatar_cell(root.frame()), controls(root.frame()), lattice(root.frame()),
    )
    for y in controls(root.frame()):
        flipped = root.clone()
        before = arr(flipped.frame()).copy()
        flipped.step(6, 3, y)
        gain = (
            0 if flipped.terminal()
            else band_shift(before, flipped.frame())
        )
        print(
            "FLIP", y, height + gain, flipped.levels_completed,
            flipped.terminal(),
            None if flipped.terminal() else avatar_cell(flipped.frame()),
            [] if flipped.terminal() else controls(flipped.frame()),
            "" if flipped.terminal() else lattice(flipped.frame()),
        )
        if flipped.levels_completed > 6 or flipped.terminal():
            continue
        for direction in ((3,), (4,)):
            walked = flipped.clone()
            walked_gain = 0
            for count in range(6):
                print(
                    "WALK", y, direction, count,
                    height + gain + walked_gain,
                    walked.levels_completed, walked.terminal(),
                    None if walked.terminal()
                    else avatar_cell(walked.frame()),
                    [] if walked.terminal()
                    else controls(walked.frame()),
                )
                if walked.levels_completed > 6 or walked.terminal():
                    break
                for row in controls(walked.frame()):
                    child = walked.clone()
                    child_gain = advance(child, [(6, 3, row)])
                    if child.levels_completed > 6:
                        print(
                            "WIN", y, direction, count, row, flush=True
                        )
                        return
                    if child_gain or child.terminal():
                        print(
                            "NEXT_FLIP", y, direction, count, row,
                            height + gain + walked_gain + child_gain,
                            child.levels_completed, child.terminal(),
                            None if child.terminal()
                            else avatar_cell(child.frame()),
                            [] if child.terminal()
                            else controls(child.frame()),
                        )
                walked_gain += advance(walked, [direction])

        ai, aj = avatar_cell(flipped.frame())
        supports = [
            click_action(i, j)
            for i in range(max(0, ai - 2), min(10, ai + 3))
            for j in range(max(0, aj - 2), min(8, aj + 3))
            if _cell_shape(flipped.frame(), i, j)[0] in (12, 14)
        ]
        outcomes = []
        for support in [None, *supports]:
            staged = flipped.clone()
            support_gain = (
                0 if support is None else advance(staged, [support])
            )
            if staged.terminal():
                continue
            for left_count in (0, 1):
                crossed = staged.clone()
                move_gain = advance(crossed, [(3,)] * left_count)
                if crossed.terminal():
                    continue
                for row in controls(crossed.frame()):
                    child = crossed.clone()
                    flip_gain = advance(child, [(6, 3, row)])
                    if child.levels_completed > 6:
                        print(
                            "SUPPORT_WIN", y, support, left_count, row,
                            flush=True,
                        )
                        return
                    if not child.terminal():
                        outcomes.append(
                            (
                                height + gain + support_gain
                                + move_gain + flip_gain,
                                support, left_count, row,
                                avatar_cell(child.frame()),
                                tuple(controls(child.frame())),
                            )
                        )
        outcomes.sort(key=lambda item: (-len(item[5]), -item[0]))
        print("SUPPORT_OUTCOMES", y, len(outcomes))
        for outcome in outcomes:
            print("SUPPORT_OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
