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
        *base[:40], click_action(7, 3), *base[40:],
        (3,), (3,), (3,), (3,),
    ]
    root = env.clone()
    height = advance(root, route)
    print(
        "ROOT", len(route), height, root.levels_completed, root.terminal(),
        avatar_cell(root.frame()), controls(root.frame()), lattice(root.frame()),
    )
    for y in controls(root.frame()):
        landed = root.clone()
        before = arr(landed.frame()).copy()
        landed.step(6, 3, y)
        gain = (
            0 if landed.terminal()
            else band_shift(before, landed.frame())
        )
        print(
            "LAND", y, height + gain, landed.levels_completed,
            landed.terminal(),
            None if landed.terminal() else avatar_cell(landed.frame()),
            [] if landed.terminal() else controls(landed.frame()),
            "" if landed.terminal() else lattice(landed.frame()),
        )
        if landed.levels_completed > 6 or landed.terminal():
            continue
        ai, aj = avatar_cell(landed.frame())
        local = [(3,), (4,)]
        local += [(6, 3, row) for row in controls(landed.frame())]
        local += [
            click_action(i, j)
            for i in range(max(0, ai - 2), min(10, ai + 3))
            for j in range(max(0, aj - 2), min(8, aj + 3))
            if _cell_shape(landed.frame(), i, j)[0] in (0, 12, 14, 15)
        ]
        for action in local:
            child = landed.clone()
            before = arr(child.frame()).copy()
            child.step(*action)
            child_gain = (
                0 if child.terminal()
                else band_shift(before, child.frame())
            )
            if (
                child.levels_completed > 6
                or child_gain
                or (
                    not child.terminal()
                    and (
                        avatar_cell(child.frame())
                        != avatar_cell(landed.frame())
                        or controls(child.frame())
                        != controls(landed.frame())
                    )
                )
            ):
                print(
                    "LOCAL", y, action, height + gain + child_gain,
                    child.levels_completed, child.terminal(),
                    None if child.terminal()
                    else avatar_cell(child.frame()),
                    [] if child.terminal()
                    else controls(child.frame()),
                )
        returned = landed.clone()
        return_gain = advance(returned, [(6, 3, 15)])
        print(
            "RETURN", y, height + gain + return_gain,
            avatar_cell(returned.frame()), controls(returned.frame()),
            lattice(returned.frame()),
        )
        walked = returned.clone()
        walked_gain = 0
        for count in range(7):
            print(
                "RIGHT_STATE", y, count,
                height + gain + return_gain + walked_gain,
                walked.levels_completed, walked.terminal(),
                None if walked.terminal() else avatar_cell(walked.frame()),
                [] if walked.terminal() else controls(walked.frame()),
            )
            if walked.terminal() or walked.levels_completed > 6:
                break
            for row in controls(walked.frame()):
                child = walked.clone()
                child_gain = advance(child, [(6, 3, row)])
                if (
                    child.levels_completed > 6
                    or child_gain
                    or child.terminal()
                    or controls(child.frame()) != controls(walked.frame())
                ):
                    print(
                        "RIGHT_SWITCH", y, count, row,
                        height + gain + return_gain + walked_gain
                        + child_gain,
                        child.levels_completed, child.terminal(),
                        None if child.terminal()
                        else avatar_cell(child.frame()),
                        [] if child.terminal()
                        else controls(child.frame()),
                    )
            walked_gain += advance(walked, [(4,)])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
