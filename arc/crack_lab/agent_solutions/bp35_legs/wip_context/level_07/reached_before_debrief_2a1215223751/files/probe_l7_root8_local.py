"""Local gravity outcomes from the first right-corridor-with-controls decode."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l7_decode_matrix import controls, target
from probe_l7_support_decode_beam import normalized_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)
OVERRIDES = {
    19: (6, 27, 27),
    21: (6, 39, 39),
    22: (6, 33, 33),
    23: (6, 33, 57),
    41: (6, 27, 33),
}


def decoded_route():
    with open("frontier_scaffold.json") as stream:
        route = normalized_route(
            json.load(stream)["staged_prefix_actions"]
        )
    overrides = OVERRIDES
    if os.environ.get("ROOT_OVERRIDES"):
        overrides = {
            int(step): tuple(action)
            for step, action in json.loads(
                os.environ["ROOT_OVERRIDES"]
            ).items()
        }
    for step, action in overrides.items():
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


def run(node, actions):
    for action in actions:
        node.step(*action)
        if node.terminal() or node.levels_completed > 6:
            break


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run(env, decoded_route())
    base_level = int(env.levels_completed)
    root = env.clone()
    print("ROOT8", summary(root), flush=True)
    outcomes = {}
    for direction in (LEFT, RIGHT):
        staged = root.clone()
        walk = []
        for count in range(7):
            if not staged.terminal():
                for control in controls(staged.frame()):
                    node = staged.clone()
                    node.step(*control)
                    route = [*walk, control]
                    if node.levels_completed > base_level:
                        print(
                            "ROOT8_WIN", [*decoded_route(), *route],
                            flush=True,
                        )
                        return
                    outcomes.setdefault(
                        summary(node), (direction, count, control)
                    )
            if staged.terminal():
                break
            staged.step(*direction)
            walk.append(direction)
    print("ROOT8_OUTCOMES", len(outcomes), flush=True)
    for state, witness in outcomes.items():
        print("ROOT8_STATE", witness, state, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
