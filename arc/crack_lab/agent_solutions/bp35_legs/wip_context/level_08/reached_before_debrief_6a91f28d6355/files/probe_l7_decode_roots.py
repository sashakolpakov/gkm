"""List distinct top roots from the finite coordinate/control decode matrix."""

import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l7_decode_matrix import build, controls, target
from probe_level7_coordinate_decode import AMBIGUOUS
from probe_level7_reward_recovery import avatar_cell, lattice


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    root = env.clone()
    unique = {}
    for flags in itertools.product((False, True), repeat=len(AMBIGUOUS)):
        for bottom in (False, True):
            node = root.clone()
            repairs = []
            route = []
            for step, action in enumerate(build(raw, flags), 1):
                candidate = action
                if (
                    len(action) == 3
                    and action[0] == 6
                    and action[1] <= 5
                    and int(node.frame()[action[2]][action[1]]) != 8
                ):
                    visible = controls(node.frame())
                    if visible:
                        candidate = visible[-1] if bottom else visible[0]
                        repairs.append((step, action, candidate))
                node.step(*candidate)
                route.append(candidate)
                if node.terminal():
                    break
            if node.terminal():
                continue
            key = np.asarray(node.frame())[:63].tobytes()
            unique.setdefault(
                key,
                (
                    flags, bottom, repairs, route,
                    avatar_cell(node.frame()), target(node.frame()),
                    tuple(controls(node.frame())), lattice(node.frame()),
                ),
            )
    print("DECODE_ROOT_COUNT", len(unique), flush=True)
    for index, item in enumerate(unique.values()):
        flags, bottom, repairs, _route, avatar, prize, switches, grid = item
        print(
            "DECODE_ROOT", index, flags, bottom, repairs,
            avatar, prize, switches, grid, flush=True,
        )


arena.run_program("bp35", probe)
