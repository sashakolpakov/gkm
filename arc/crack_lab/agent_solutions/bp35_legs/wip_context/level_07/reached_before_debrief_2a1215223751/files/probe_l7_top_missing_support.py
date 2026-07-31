"""Test one omitted top-room support before the preserved final suffix."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l7_decode_matrix import controls
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)
    for action in decoded_route():
        env.step(*action)
    root = env.clone()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    print(
        "TOP_MISSING_ROOT", avatar_cell(root.frame()),
        tuple(controls(root.frame())), len(supports), lattice(root.frame()),
        flush=True,
    )
    outcomes = {}
    for support in supports:
        for direction in (LEFT, RIGHT):
            node = root.clone()
            suffix = [support, *([direction] * 4)]
            for action in suffix:
                node.step(*action)
                if node.terminal() or node.levels_completed > base_level:
                    break
            if node.levels_completed > base_level:
                print(
                    "TOP_MISSING_WIN", [*decoded_route(), *suffix],
                    flush=True,
                )
                return
            if node.terminal():
                continue
            for control in controls(node.frame()):
                child = node.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "TOP_MISSING_WIN",
                        [*decoded_route(), *suffix, control], flush=True,
                    )
                    return
                if not child.terminal():
                    key = (
                        avatar_cell(child.frame()),
                        tuple(controls(child.frame())),
                        lattice(child.frame()),
                    )
                    outcomes.setdefault(key, (support, direction, control))
    print("TOP_MISSING_OUTCOMES", len(outcomes), flush=True)
    for state, witness in outcomes.items():
        print("TOP_MISSING_STATE", witness, state, flush=True)


arena.run_program("bp35", probe)
