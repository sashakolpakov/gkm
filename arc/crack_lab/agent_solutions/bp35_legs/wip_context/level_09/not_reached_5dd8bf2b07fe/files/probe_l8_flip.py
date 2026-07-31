"""Flip the newly exposed top control and compare downward landings."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import avatar_column, band_shift, click_action
from perception import arr
from probe_l8_stage1 import lattice, target
from probe_l8_top import TOP_ROUTE


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


FLIP = click_action(0, 3)


def route_to_column(column):
    route = []
    for next_col in range(2, column + 1):
        route.extend([click_action(6, next_col), (4,)])
    route.extend([click_action(5, column), click_action(5, column)])
    return route


def run(node, route, trace=False):
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if trace:
            up = 0 if node.terminal() else band_shift(before, node.frame())
            down = 0 if node.terminal() else band_shift(node.frame(), before)
            print(
                "STEP",
                index,
                action,
                "alive",
                not node.terminal(),
                "level",
                node.levels_completed,
                "up",
                up,
                "down",
                down,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for column in (2, 3, 4):
        node = env.clone()
        route = [*TOP_ROUTE, *route_to_column(column)]
        run(node, route)
        before = arr(node.frame()).copy()
        node.step(*FLIP)
        print(
            "CASE",
            column,
            "pre",
            lattice(before),
            "post",
            node.levels_completed,
            node.terminal(),
            0 if node.terminal() else band_shift(before, node.frame()),
            0 if node.terminal() else band_shift(node.frame(), before),
            None if node.terminal() else avatar_column(node.frame()),
            None if node.terminal() else target(node.frame()),
            "" if node.terminal() else lattice(node.frame()),
        )

    print("DIRECT")
    route = [*TOP_ROUTE, *route_to_column(3), FLIP]
    run(env, route, trace=True)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
