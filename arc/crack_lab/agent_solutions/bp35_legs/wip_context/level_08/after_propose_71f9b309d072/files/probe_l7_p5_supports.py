"""Single-support setups at the deepest support-free macro landing."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_root8_local import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)
P5 = [
    RIGHT, RIGHT, (6, 3, 21),
    LEFT, (6, 3, 0), (6, 3, 57),
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
    run_actions(env, [*decoded_route(), *P5])
    base_level = int(env.levels_completed)
    root = env.clone()
    root_frame = root.frame()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(root_frame, i, j)[0] in (12, 14)
    ]
    print(
        "P5_ROOT", summary(root), "supports", supports, flush=True,
    )
    effects = {}
    for action in [LEFT, RIGHT, *controls(root_frame), *supports]:
        node = root.clone()
        node.step(*action)
        effects.setdefault(summary(node), action)
    print("P5_EFFECTS", len(effects), flush=True)
    for state, action in effects.items():
        print("P5_ONE", action, state, flush=True)

    outcomes = {}
    best_delta = -99
    for setup in [None, *supports]:
        for direction in (LEFT, RIGHT):
            staged = root.clone()
            prefix = [] if setup is None else [setup]
            if setup is not None:
                staged.step(*setup)
            for count in range(4):
                if not staged.terminal():
                    for control in controls(staged.frame()):
                        node = staged.clone()
                        before = node.frame()
                        node.step(*control)
                        route = [*prefix, *([direction] * count), control]
                        if node.levels_completed > base_level:
                            print(
                                "P5_WIN",
                                [*decoded_route(), *P5, *route],
                                flush=True,
                            )
                            return
                        if node.terminal():
                            continue
                        delta = signed_origin_delta(before, node.frame())
                        key = summary(node)
                        outcomes.setdefault(
                            key, (setup, direction, count, control, delta)
                        )
                        if delta > best_delta:
                            best_delta = delta
                            print(
                                "P5_PROGRESS", best_delta,
                                (setup, direction, count, control),
                                key, flush=True,
                            )
                if staged.terminal():
                    break
                staged.step(*direction)
    print("P5_OUTCOMES", len(outcomes), best_delta, flush=True)
    for state, witness in outcomes.items():
        print("P5_STATE", witness, state, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
