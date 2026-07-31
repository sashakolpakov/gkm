import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from perception import connected_components
from probe_level7_coordinate_decode import advance
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_items = json.load(stream)["staged_prefix_actions"]
    raw_route = [
        (item,) if isinstance(item, int) else tuple(item)
        for item in raw_items
    ]
    supports = [
        (i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] == 12
    ]
    outcomes = []
    for support in supports:
        route = [
            click_action(*support), *raw_route,
            (3,), (3,), (3,), (3,),
        ]
        node = env.clone()
        height = advance(node, route)
        if node.levels_completed > 6:
            print("WIN_BEFORE", support, route, flush=True)
            return
        if node.terminal():
            outcomes.append((False, height, support, None, ()))
            continue
        blobs = connected_components(node.frame(), colors=(8,), min_area=1)
        for blob in blobs:
            r0, c0, r1, c1 = blob.bbox
            child = node.clone()
            child.step(6, (c0 + c1) // 2, (r0 + r1) // 2)
            if child.levels_completed > 6:
                print(
                    "LEADING_WIN", support, blob.bbox,
                    [
                        *route,
                        (6, (c0 + c1) // 2, (r0 + r1) // 2),
                    ],
                    flush=True,
                )
                return
        outcomes.append(
            (
                True, height, support, avatar_cell(node.frame()),
                tuple(controls(node.frame())),
            )
        )
    outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[4]))
    )
    print("NO_LEADING_WIN", len(outcomes))
    for outcome in outcomes:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
