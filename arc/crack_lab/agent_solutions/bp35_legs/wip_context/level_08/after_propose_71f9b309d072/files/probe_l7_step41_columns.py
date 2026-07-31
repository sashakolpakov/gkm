"""Resolve the route's decisive step-41 support across every visible column."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, _cell_shape
from probe_l7_decode_matrix import controls, target
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)
    route = decoded_route()
    for action in route[:40]:
        env.step(*action)
    print(
        "STEP41_ROOT", avatar_cell(env.frame()), tuple(controls(env.frame())),
        tuple((j, _cell_shape(env.frame(), 5, j)) for j in range(8)),
        lattice(env.frame()), flush=True,
    )
    outcomes = {}
    for x in COL_ANCHORS:
        for repair_mode in ("none", "near", "top", "bottom"):
            node = env.clone()
            candidate_route = [(6, x, 33)]
            node.step(6, x, 33)
            repairs = []
            for step, action in enumerate(route[41:], 42):
                candidate = action
                if (
                    repair_mode != "none"
                    and len(action) == 3
                    and action[0] == 6
                    and action[1] <= 5
                    and int(node.frame()[action[2]][action[1]]) != 8
                ):
                    visible = controls(node.frame())
                    if visible:
                        if repair_mode == "near":
                            candidate = min(
                                visible,
                                key=lambda item: abs(item[2] - action[2]),
                            )
                        elif repair_mode == "top":
                            candidate = visible[0]
                        else:
                            candidate = visible[-1]
                        repairs.append((step, action, candidate))
                node.step(*candidate)
                candidate_route.append(candidate)
                if node.terminal() or node.levels_completed > base_level:
                    break
            if node.levels_completed > base_level:
                print(
                    "STEP41_WIN", x, repair_mode, repairs,
                    [*route[:40], *candidate_route], flush=True,
                )
                return
            if node.terminal():
                continue
            key = (
                avatar_cell(node.frame()), target(node.frame()),
                tuple(controls(node.frame())), lattice(node.frame()),
            )
            outcomes.setdefault(key, (x, repair_mode, repairs, candidate_route))
    print("STEP41_OUTCOMES", len(outcomes), flush=True)
    for index, (state, witness) in enumerate(outcomes.items()):
        print("STEP41_STATE", index, witness[:3], state, flush=True)
        for direction in (LEFT, RIGHT):
            node = env.clone()
            run = [*witness[3], *([direction] * 6)]
            for action in run:
                node.step(*action)
                if node.terminal() or node.levels_completed > base_level:
                    break
            if node.levels_completed > base_level:
                print(
                    "STEP41_WIN", witness[:3], direction,
                    [*route[:40], *run], flush=True,
                )
                return
            if node.terminal():
                continue
            for control in controls(node.frame()):
                child = node.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "STEP41_WIN", witness[:3], direction, control,
                        [*route[:40], *run, control], flush=True,
                    )
                    return


arena.run_program("bp35", probe)
