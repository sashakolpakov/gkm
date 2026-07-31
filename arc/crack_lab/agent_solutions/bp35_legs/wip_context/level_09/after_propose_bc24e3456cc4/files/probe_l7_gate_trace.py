"""Trace the support-conditioned transition out of the hybrid top room."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l7_decode_matrix import controls, target
from probe_l7_step41_local import hybrid_route
from probe_level7_reward_recovery import avatar_cell, lattice


RIGHT = (4,)
SETUP = click_action(2, 3)


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
    for action in hybrid_route():
        env.step(*action)
    route = [SETUP, (6, 3, 35), *([RIGHT] * 6)]
    for index, action in enumerate(route, 1):
        env.step(*action)
        print("GATE_TRACE", index, action, summary(env), flush=True)
        if env.terminal() or env.levels_completed > 6:
            return
    root = env.clone()
    effects = {}
    for i in range(10):
        for j in range(8):
            if _cell_shape(root.frame(), i, j)[0] not in (12, 14):
                continue
            action = click_action(i, j)
            node = root.clone()
            node.step(*action)
            effects.setdefault(summary(node), action)
    for action in ((3,), (4,), *controls(root.frame())):
        node = root.clone()
        node.step(*action)
        effects.setdefault(summary(node), action)
    print("GATE_EFFECTS", len(effects), flush=True)
    for state, action in effects.items():
        print("GATE_ONE", action, state, flush=True)


arena.run_program("bp35", probe)
