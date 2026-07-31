import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS
from probe_level7_coordinate_decode import advance
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


EARLY = (1, 2, 3, 4, 8, 15)
FIXED_SHIFTED = {22}


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    outcomes = []
    for shifted_early in itertools.product((False, True), repeat=len(EARLY)):
        shifted_steps = FIXED_SHIFTED | {
            step
            for step, shifted in zip(EARLY, shifted_early)
            if shifted
        }
        route = []
        for step, item in enumerate(raw_route, 1):
            action = (item,) if isinstance(item, int) else tuple(item)
            if (
                step in shifted_steps
                and len(action) == 3
                and action[1] != 3
            ):
                action = (action[0], action[1] + 12, action[2])
            route.append(action)

        node = env.clone()
        height = advance(node, [*route, (3,), (3,), (3,), (3,)])
        if node.levels_completed > 6:
            print("WIN_BEFORE_SWITCH", shifted_early, route, flush=True)
            return
        if node.terminal():
            outcomes.append((False, height, shifted_early, None, ()))
            continue
        for y in controls(node.frame()):
            child = node.clone()
            child.step(6, 3, y)
            if child.levels_completed > 6:
                print(
                    "EARLY_WIN", shifted_early, y,
                    [*route, (3,), (3,), (3,), (3,), (6, 3, y)],
                    flush=True,
                )
                return
        outcomes.append(
            (
                True, height, shifted_early, avatar_cell(node.frame()),
                tuple(controls(node.frame())),
            )
        )
    outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[4]), item[2])
    )
    print("NO_EARLY_WIN", len(outcomes))
    for outcome in outcomes:
        print("EARLY", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
