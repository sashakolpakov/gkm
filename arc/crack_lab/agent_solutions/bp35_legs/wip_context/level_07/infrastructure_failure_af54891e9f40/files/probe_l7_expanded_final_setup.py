"""Test the preserved one-action final setup across expanded route decodes."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from probe_l7_decode_matrix import controls, target
from probe_l7_support_decode_beam import normalized_route, support_candidates
from probe_level7_coordinate_decode import AMBIGUOUS
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT, UNDO = (3,), (4,), (7,)


def setup_actions(frame):
    """Visible support presses plus cheap no-op/action alternatives."""
    supports = [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
    ]
    return list(dict.fromkeys([*supports, *controls(frame), LEFT, RIGHT, UNDO]))


def finish(node, prefix, base_level):
    for setup in setup_actions(node.frame()):
        child = node.clone()
        suffix = [setup, LEFT, LEFT, LEFT, LEFT]
        for action in suffix:
            child.step(*action)
            if child.terminal() or child.levels_completed > base_level:
                break
        if child.levels_completed > base_level:
            print("EXPANDED_FINAL_WIN", [*prefix, *suffix], flush=True)
            return True
        if child.terminal():
            continue
        switches = controls(child.frame())
        if not switches:
            continue
        control = switches[-1]
        child.step(*control)
        if child.levels_completed > base_level:
            print(
                "EXPANDED_FINAL_WIN",
                [*prefix, *suffix, control],
                flush=True,
            )
            return True
    return False


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    route = normalized_route(raw)
    base_level = int(env.levels_completed)
    branches = [(env.clone(), [])]

    for step, action in enumerate(route, 1):
        if step in AMBIGUOUS:
            children = []
            for node, witness in branches:
                for candidate in support_candidates(node.frame(), action):
                    child = node.clone()
                    child.step(*candidate)
                    if not child.terminal():
                        children.append((child, [*witness, candidate]))
            unique = {}
            for child, witness in children:
                key = np.asarray(child.frame())[:63].tobytes()
                unique.setdefault(key, (child, witness))
            branches = list(unique.values())
            print("EXPANDED_FINAL_BOUNDARY", step, len(branches), flush=True)
        else:
            live = []
            for node, witness in branches:
                node.step(*action)
                if not node.terminal():
                    live.append((node, [*witness, action]))
            branches = live
        if not branches:
            print("EXPANDED_FINAL_ALL_DEAD", step, flush=True)
            return

    print("EXPANDED_FINAL_ROOTS", len(branches), flush=True)
    shard = int(os.environ.get("SHARD", "0"))
    shards = int(os.environ.get("SHARDS", "1"))
    for index, (node, witness) in enumerate(branches):
        if index % shards != shard:
            continue
        print(
            "EXPANDED_FINAL_ROOT", index, avatar_cell(node.frame()),
            target(node.frame()), tuple(controls(node.frame())),
            len(setup_actions(node.frame())), lattice(node.frame()),
            flush=True,
        )
        if finish(node, witness, base_level):
            return
    print("EXPANDED_FINAL_DONE", len(branches), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
