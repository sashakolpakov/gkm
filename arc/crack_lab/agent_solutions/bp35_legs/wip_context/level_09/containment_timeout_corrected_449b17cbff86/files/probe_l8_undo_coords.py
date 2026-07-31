"""Determine whether action 7 is coordinate-addressed or strictly LIFO."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, click_action
from probe_l8_stage1 import lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


A_CLICK = click_action(3, 0)
B_CLICK = click_action(3, 1)


def run(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def summary(node):
    return (
        node.levels_completed,
        node.terminal(),
        None if node.terminal() else avatar_column(node.frame()),
        None if node.terminal() else _cell_shape(node.frame(), 3, 0),
        None if node.terminal() else _cell_shape(node.frame(), 3, 1),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    variants = {
        "ab_plain": [A_CLICK, B_CLICK, (7,)],
        "ab_release_a": [A_CLICK, B_CLICK, (7, A_CLICK[1], A_CLICK[2])],
        "ab_release_b": [A_CLICK, B_CLICK, (7, B_CLICK[1], B_CLICK[2])],
        "ab_release_empty": [A_CLICK, B_CLICK, (7, 57, 57)],
        "a_move_plain": [A_CLICK, (4,), (7,)],
        "a_move_release_a": [A_CLICK, (4,), (7, A_CLICK[1], A_CLICK[2])],
        "a_move_release_empty": [A_CLICK, (4,), (7, 57, 57)],
    }
    for label, route in variants.items():
        node = env.clone()
        run(node, route)
        print("CASE", label, route, summary(node))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
