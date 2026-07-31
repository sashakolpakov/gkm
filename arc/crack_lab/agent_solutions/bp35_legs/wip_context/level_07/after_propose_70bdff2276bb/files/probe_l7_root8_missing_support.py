"""One omitted support at corrected root 8 plus the preserved final suffix."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_root8_local import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def summary(node):
    if node.terminal():
        return ("dead", int(node.levels_completed))
    return (
        "alive", int(node.levels_completed), avatar_cell(node.frame()),
        target(node.frame()), tuple(controls(node.frame())),
        lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run_actions(env, decoded_route())
    base_level = int(env.levels_completed)
    root = env.clone()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    print(
        "ROOT8_MISSING_ROOT", len(supports), summary(root), flush=True,
    )
    outcomes = {}
    for support in supports:
        for direction in (LEFT, RIGHT):
            node = root.clone()
            suffix = [support, *([direction] * 4)]
            run_actions(node, suffix)
            if node.levels_completed > base_level:
                print(
                    "ROOT8_MISSING_WIN",
                    [*decoded_route(), *suffix], flush=True,
                )
                return
            if node.terminal():
                continue
            for control in controls(node.frame()):
                child = node.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "ROOT8_MISSING_WIN",
                        [*decoded_route(), *suffix, control], flush=True,
                    )
                    return
                if not child.terminal():
                    outcomes.setdefault(
                        summary(child), (support, direction, control)
                    )
    print("ROOT8_MISSING_DONE", len(outcomes), flush=True)
    for state, witness in outcomes.items():
        print("ROOT8_MISSING_STATE", witness, state, flush=True)


arena.run_program("bp35", probe)
