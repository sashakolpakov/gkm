"""Test the forced top-right commit at each helper patrol phase."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


ROOT = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
] + ([2, 1] * 7)[:13] + [1, 3, 3, 1, 1, 5]


def apply(node, actions):
    child = node.clone()
    for action in actions:
        if child.terminal():
            break
        child.step(action)
    return child


def helper_row(node):
    blobs = connected_components(node.frame(), colors=(14,), min_area=4)
    return None if not blobs else blobs[0].bbox[0]


def marker(node):
    blobs = connected_components(node.frame(), colors=(9,), min_area=4)
    blob = next((b for b in blobs if b.bbox[0] == 1), None)
    return None if blob is None else blob.bbox[:2]


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + ROOT:
        env.step(action)
    base = int(env.levels_completed)
    wait = []
    for phase in range(16):
        phased = apply(env, wait)
        _, reach = fast_reach(phased)
        path = reach.get((2, 50))
        if path is None:
            wait.append(2 if phase % 2 == 0 else 1)
            continue
        child = apply(phased, path + [5])
        valid = marker(child) != marker(phased)
        minimum = helper_row(child)
        min_barrier = int(np.count_nonzero(np.asarray(child.frame()) == 15))
        won = int(child.levels_completed) > base
        for tick in range(24):
            if child.terminal():
                break
            child.step(2 if tick % 2 == 0 else 1)
            row = helper_row(child)
            if row is not None:
                minimum = row if minimum is None else min(minimum, row)
            min_barrier = min(
                min_barrier,
                int(np.count_nonzero(np.asarray(child.frame()) == 15)),
            )
            won = won or int(child.levels_completed) > base
        reward_path, child_reach = fast_reach(child)
        print(
            "case", phase, tuple(path + [5]), "valid", valid,
            "helper", helper_row(child), "minimum", minimum,
            "barrier", min_barrier, "reach", len(child_reach),
            "win", reward_path, won,
            "frontier", tuple(
                (pos, len(walk))
                for pos, walk in _special_frontier(
                    child_reach, child.frame()
                )
            ),
            flush=True,
        )
        wait.append(2 if phase % 2 == 0 else 1)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
