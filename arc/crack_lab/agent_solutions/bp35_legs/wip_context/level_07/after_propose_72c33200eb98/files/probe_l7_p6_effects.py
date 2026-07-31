"""Distinct effects in the lower corridor reached from the P5 left gate."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_p5_supports import P5
from probe_l7_root8_local import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)
P6 = [click_action(4, 2), LEFT, (6, 3, 41)]


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
    run_actions(env, [*decoded_route(), *P5, *P6])
    root = env.clone()
    print("P6_ROOT", summary(root), flush=True)
    actions = [LEFT, RIGHT, *controls(root.frame())]
    for i in range(10):
        for j in range(8):
            if _cell_shape(root.frame(), i, j)[0] in (12, 14):
                actions.append(click_action(i, j))
    effects = {}
    for action in actions:
        node = root.clone()
        node.step(*action)
        effects.setdefault(summary(node), action)
    print("P6_EFFECTS", len(effects), flush=True)
    for state, action in effects.items():
        print("P6_ONE", action, state, flush=True)


arena.run_program("bp35", probe)
