"""Track each high-control root support through the eight-band macro."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_s8_supports import TO_S8, decoded_route
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
    route = decoded_route()
    run_actions(env, route)
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
        run_actions(node, [support, *TO_S8])
        if node.levels_completed > base_level:
            print(
                "ROOTD_STAGE_WIN",
                [*route, support, *TO_S8], flush=True,
            )
            return
        outcomes.setdefault(summary(node), support)
    print("ROOTD_STAGE_S8", len(outcomes), flush=True)
    for state, support in outcomes.items():
        print("ROOTD_STAGE_STATE", support, state, flush=True)


arena.run_program("bp35", probe)
