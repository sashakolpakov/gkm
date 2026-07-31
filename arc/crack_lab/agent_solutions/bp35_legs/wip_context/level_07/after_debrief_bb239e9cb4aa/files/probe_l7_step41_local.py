"""Symbolic one- and two-action affordances in the step-41 hybrid room."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l7_decode_matrix import controls, target
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def summary(node):
    if node.terminal():
        return ("dead", int(node.levels_completed))
    return (
        "alive", int(node.levels_completed), avatar_cell(node.frame()),
        target(node.frame()), tuple(controls(node.frame())),
        lattice(node.frame()),
    )


def hybrid_route():
    route = decoded_route()
    route[40] = (6, 39, 33)
    return route


def run(node, actions):
    for action in actions:
        node.step(*action)
        if node.terminal() or node.levels_completed > 6:
            break


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run(env, hybrid_route())
    print("STEP41_LOCAL_ROOT", summary(env), flush=True)
    root = env.clone()
    frame = root.frame()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
    ]
    first = [LEFT, RIGHT, *controls(frame), *supports]
    effects = {}
    for action in first:
        node = root.clone()
        run(node, [action])
        effects.setdefault(summary(node), action)
    print("STEP41_LOCAL_EFFECTS", len(effects), flush=True)
    for state, action in effects.items():
        print("STEP41_LOCAL_ONE", action, state, flush=True)

    outcomes = {}
    for setup in [None, *supports]:
        for movement in ([], [LEFT], [RIGHT], [LEFT, LEFT], [RIGHT, RIGHT]):
            staged = root.clone()
            prefix = ([] if setup is None else [setup]) + movement
            run(staged, prefix)
            if staged.terminal():
                continue
            for control in controls(staged.frame()):
                node = staged.clone()
                run(node, [control])
                route = [*prefix, control]
                if node.levels_completed > 6:
                    print(
                        "STEP41_LOCAL_WIN", [*hybrid_route(), *route],
                        flush=True,
                    )
                    return
                if node.terminal():
                    continue
                for direction in (LEFT, RIGHT):
                    walked = node.clone()
                    suffix = []
                    for _ in range(6):
                        run(walked, [direction])
                        suffix.append(direction)
                        if walked.levels_completed > 6:
                            print(
                                "STEP41_LOCAL_WIN",
                                [*hybrid_route(), *route, *suffix],
                                flush=True,
                            )
                            return
                        if walked.terminal():
                            break
                    if not walked.terminal():
                        outcomes.setdefault(
                            summary(walked),
                            (setup, tuple(movement), control, direction),
                        )
    print("STEP41_LOCAL_OUTCOMES", len(outcomes), flush=True)
    for state, witness in outcomes.items():
        print("STEP41_LOCAL_STATE", witness, state, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
