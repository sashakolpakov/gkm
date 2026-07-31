"""Distinct effects in the right corridor reached through the center gate."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l7_decode_matrix import controls, target
from probe_l7_step41_local import hybrid_route
from probe_level7_reward_recovery import avatar_cell, lattice


PREFIX = [
    click_action(2, 3),
    (6, 3, 35),
    click_action(5, 3),
]


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
    for action in [*hybrid_route(), *PREFIX]:
        env.step(*action)
    print("CORRIDOR_ROOT", summary(env), flush=True)
    root = env.clone()
    actions = [(3,), (4,), *controls(root.frame())]
    for i in range(10):
        for j in range(8):
            if _cell_shape(root.frame(), i, j)[0] in (12, 14):
                actions.append(click_action(i, j))
    effects = {}
    for action in actions:
        node = root.clone()
        node.step(*action)
        effects.setdefault(summary(node), action)
    print("CORRIDOR_EFFECTS", len(effects), flush=True)
    for state, action in effects.items():
        print("CORRIDOR_ONE", action, state, flush=True)


arena.run_program("bp35", probe)
