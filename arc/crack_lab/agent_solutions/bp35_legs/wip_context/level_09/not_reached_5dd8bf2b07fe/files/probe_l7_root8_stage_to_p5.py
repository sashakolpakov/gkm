"""Track each root-8 support through the known five-band macro."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_p5_supports import P5
from probe_l7_root8_local import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


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
    outcomes = {}
    for support in supports:
        node = root.clone()
        run_actions(node, [support, *P5])
        if node.levels_completed > base_level:
            print(
                "ROOT8_STAGE_WIN",
                [*decoded_route(), support, *P5], flush=True,
            )
            return
        outcomes.setdefault(summary(node), support)
    print("ROOT8_STAGE_P5", len(outcomes), flush=True)
    for state, support in outcomes.items():
        print("ROOT8_STAGE_STATE", support, state, flush=True)


arena.run_program("bp35", probe)
