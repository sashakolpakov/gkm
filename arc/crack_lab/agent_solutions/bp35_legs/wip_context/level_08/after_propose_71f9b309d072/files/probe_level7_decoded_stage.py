import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


SHIFTED_STEPS = {1, 2, 3, 4, 8, 15, 22}


def decoded_route():
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]
    route = []
    for step, item in enumerate(raw_route, 1):
        action = (item,) if isinstance(item, int) else tuple(item)
        if (
            step in SHIFTED_STEPS
            and len(action) == 3
            and action[1] != 3
        ):
            action = (action[0], action[1] + 12, action[2])
        route.append(action)
    return route


def support_actions(frame):
    return [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def try_finish(env, route):
    node = env.clone()
    height = advance(node, [*route, (3,), (3,), (3,), (3,)])
    if node.levels_completed > 6:
        return True, height, None, node
    if node.terminal():
        return False, height, None, node
    for y in controls(node.frame()):
        child = node.clone()
        child.step(6, 3, y)
        if child.levels_completed > 6:
            return True, height, y, child
    return False, height, None, node


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = decoded_route()
    outcomes = []
    for boundary in (40, 41):
        stage = env.clone()
        advance(stage, route[:boundary])
        candidates = support_actions(stage.frame())
        print(
            "BOUNDARY", boundary, avatar_cell(stage.frame()),
            controls(stage.frame()),
            [
                (
                    action,
                    _cell_shape(
                        stage.frame(),
                        (action[2] - 3) // 6,
                        (action[1] - 15) // 6,
                    ),
                )
                for action in candidates
            ],
            lattice(stage.frame()),
        )
        for action in candidates:
            candidate_route = [
                *route[:boundary], action, *route[boundary:]
            ]
            won, height, switch, node = try_finish(env, candidate_route)
            if won:
                print(
                    "STAGE_WIN", boundary, action, switch,
                    [
                        *candidate_route,
                        (3,), (3,), (3,), (3,),
                        *([] if switch is None else [(6, 3, switch)]),
                    ],
                    flush=True,
                )
                return
            outcomes.append(
                (
                    height, boundary, action,
                    None if node.terminal() else avatar_cell(node.frame()),
                    () if node.terminal() else tuple(controls(node.frame())),
                )
            )
    outcomes.sort(key=lambda item: (-item[0], -len(item[4])))
    print("NO_STAGE_WIN", len(outcomes))
    for outcome in outcomes:
        print("OPTION", outcome)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
