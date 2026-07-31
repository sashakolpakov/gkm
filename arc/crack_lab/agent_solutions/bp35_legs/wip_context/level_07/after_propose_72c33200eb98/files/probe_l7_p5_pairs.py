"""Two-support staging at P5 before its sole gravity control."""

import itertools
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_p5_supports import P5
from probe_l7_root8_local import decoded_route
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


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run_actions(env, [*decoded_route(), *P5])
    base_level = int(env.levels_completed)
    root = env.clone()
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    pairs = list(itertools.combinations(supports, 2))
    print("P5_PAIRS_ROOT", len(pairs), supports, flush=True)
    outcomes = {}
    best = (-99, -1)
    for pair in pairs:
        staged_pair = root.clone()
        run_actions(staged_pair, pair)
        for direction in (LEFT, RIGHT):
            staged = staged_pair.clone()
            for count in range(4):
                if staged.terminal():
                    break
                for control in controls(staged.frame()):
                    child = staged.clone()
                    before = child.frame()
                    child.step(*control)
                    route = [*pair, *([direction] * count), control]
                    if child.levels_completed > base_level:
                        print(
                            "P5_PAIRS_WIN",
                            [*decoded_route(), *P5, *route], flush=True,
                        )
                        return
                    if child.terminal():
                        continue
                    delta = signed_origin_delta(before, child.frame())
                    child_summary = summary(child)
                    outcomes.setdefault(
                        child_summary,
                        (pair, direction, count, control, delta),
                    )
                    rank = (delta, len(controls(child.frame())))
                    if rank > best:
                        best = rank
                        print(
                            "P5_PAIRS_PROGRESS", best,
                            (pair, direction, count, control),
                            child_summary, flush=True,
                        )
                staged.step(*direction)
    print("P5_PAIRS_DONE", len(outcomes), best, flush=True)
    for state, witness in outcomes.items():
        if state[3] is not None or state[4] or witness[-1] > 1:
            print("P5_PAIRS_STATE", witness, state, flush=True)


arena.run_program("bp35", probe)
