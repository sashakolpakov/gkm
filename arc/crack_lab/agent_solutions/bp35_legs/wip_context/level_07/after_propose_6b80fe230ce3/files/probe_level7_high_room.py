import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_decode_frontiers import build_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


VARIANTS = [
    (False, False, True, False, True),
    (True, False, True, False, True),
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    outcomes = []
    for late in VARIANTS:
        root = env.clone()
        route = [
            *build_route(raw_route, late), (3,), (3,), (3,), (3,)
        ]
        height = advance(root, route)
        print(
            "ROOT", late, height, avatar_cell(root.frame()),
            controls(root.frame()),
            (4, 2, _cell_shape(root.frame(), 4, 2)),
            (4, 4, _cell_shape(root.frame(), 4, 4)),
            lattice(root.frame()),
        )
        for support in (None, (4, 2), (4, 4)):
            staged = root.clone()
            support_path = (
                [] if support is None else [click_action(*support)]
            )
            support_gain = advance(staged, support_path)
            if staged.terminal():
                continue
            for direction in ((3,), (4,)):
                walked = staged.clone()
                walked_gain = 0
                for count in range(4):
                    if walked.levels_completed > 6:
                        print(
                            "WIN_WALK", late, support, direction, count,
                            flush=True,
                        )
                        return
                    if walked.terminal():
                        break
                    for y in controls(walked.frame()):
                        child = walked.clone()
                        flip_gain = advance(child, [(6, 3, y)])
                        if child.levels_completed > 6:
                            print(
                                "WIN_FLIP", late, support, direction,
                                count, y, flush=True,
                            )
                            return
                        if not child.terminal():
                            outcomes.append(
                                (
                                    height + support_gain + walked_gain
                                    + flip_gain,
                                    late, support, direction, count, y,
                                    avatar_cell(child.frame()),
                                    tuple(controls(child.frame())),
                                    lattice(child.frame()),
                                )
                            )
                    walked_gain += advance(walked, [direction])
    outcomes.sort(
        key=lambda item: (-item[0], -len(item[7])),
    )
    print("NO_HIGH_ROOM_WIN", len(outcomes))
    for outcome in outcomes:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
