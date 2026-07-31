"""Single-support setups in the deepest high-control decoded room."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_support_decode_beam import normalized_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)
OVERRIDES = {
    19: (6, 45, 27),
    21: (6, 27, 39),
    22: (6, 15, 33),
    23: (6, 33, 57),
    41: (6, 27, 33),
}
TO_S8 = [
    RIGHT, RIGHT, (6, 3, 21),
    LEFT, (6, 3, 35), (6, 3, 9),
]


def decoded_route():
    with open("frontier_scaffold.json") as stream:
        route = normalized_route(
            json.load(stream)["staged_prefix_actions"]
        )
    for step, action in OVERRIDES.items():
        route[step - 1] = action
    return route


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
    staged_action = None
    if os.environ.get("STAGE_ACTION"):
        staged_action = tuple(json.loads(os.environ["STAGE_ACTION"]))
    run_actions(
        env,
        [
            *route,
            *([] if staged_action is None else [staged_action]),
            *TO_S8,
        ],
    )
    base_level = int(env.levels_completed)
    root = env.clone()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    print("S8_ROOT", summary(root), supports, flush=True)
    outcomes = {}
    best = (-99, -1)
    for setup in [None, *supports]:
        for direction in (LEFT, RIGHT):
            staged = root.clone()
            prefix = [] if setup is None else [setup]
            if setup is not None:
                staged.step(*setup)
            for count in range(4):
                if staged.terminal():
                    break
                for control in controls(staged.frame()):
                    child = staged.clone()
                    before = child.frame()
                    child.step(*control)
                    suffix = [*prefix, *([direction] * count), control]
                    if child.levels_completed > base_level:
                        print(
                            "S8_WIN",
                            [
                                *route,
                                *(
                                    []
                                    if staged_action is None
                                    else [staged_action]
                                ),
                                *TO_S8,
                                *suffix,
                            ],
                            flush=True,
                        )
                        return
                    if child.terminal():
                        continue
                    delta = signed_origin_delta(before, child.frame())
                    state = summary(child)
                    outcomes.setdefault(
                        state, (setup, direction, count, control, delta)
                    )
                    rank = (delta, len(controls(child.frame())))
                    if rank > best:
                        best = rank
                        print(
                            "S8_PROGRESS", best,
                            (setup, direction, count, control),
                            state, flush=True,
                        )
                staged.step(*direction)
    print("S8_DONE", len(outcomes), best, flush=True)
    for state, witness in outcomes.items():
        print("S8_STATE", witness, state, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
