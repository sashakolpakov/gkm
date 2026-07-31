import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from probe_level7_coordinate_decode import advance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = decoded_route()
    outcomes = []
    checked = 0
    for boundary in range(len(route) + 1):
        for movement in ((3,), (4,)):
            checked += 1
            candidate = [
                *route[:boundary], movement, *route[boundary:]
            ]
            node = env.clone()
            height = advance(
                node, [*candidate, (3,), (3,), (3,), (3,)]
            )
            if node.levels_completed > 6:
                print(
                    "MOVE_WIN_BEFORE", boundary, movement, candidate,
                    flush=True,
                )
                return
            if node.terminal():
                outcomes.append(
                    (False, height, boundary, movement, None, ())
                )
                continue
            for y in controls(node.frame()):
                child = node.clone()
                child.step(6, 3, y)
                if child.levels_completed > 6:
                    print(
                        "MOVE_WIN", boundary, movement, y,
                        [
                            *candidate,
                            (3,), (3,), (3,), (3,), (6, 3, y),
                        ],
                        flush=True,
                    )
                    return
            outcomes.append(
                (
                    True, height, boundary, movement,
                    avatar_cell(node.frame()), tuple(controls(node.frame())),
                )
            )
            if checked % 40 == 0:
                print("CHECKED", checked, flush=True)
    outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[5]))
    )
    print("NO_MOVE_WIN", len(outcomes))
    for outcome in outcomes[:50]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
