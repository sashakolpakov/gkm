"""Bounded phase/commit table from the reproduced level-7 open-gate branch."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _avatar_pos, _special_frontier, fast_reach
from perception import connected_components


OPEN = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
]


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


def barrier_area(node):
    return int(np.count_nonzero(np.asarray(node.frame()) == 15))


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + OPEN:
        env.step(action)
    base = int(env.levels_completed)
    rows = []
    wait = []
    for phase in range(16):
        phased = apply(env, wait)
        reward_path, reach = fast_reach(phased)
        targets = {(26, 26), (32, 26)}
        targets.update(pos for pos, _ in _special_frontier(
            reach, phased.frame()
        ))
        for pos in sorted(targets):
            path = reach.get(pos)
            if path is None:
                continue
            child = apply(phased, path + [5])
            if marker(child) == marker(phased):
                continue
            minimum = helper_row(child)
            min_barrier = barrier_area(child)
            won = int(child.levels_completed) > base
            for tick in range(20):
                if child.terminal():
                    break
                child.step(2 if tick % 2 == 0 else 1)
                row = helper_row(child)
                if row is not None:
                    minimum = row if minimum is None else min(minimum, row)
                min_barrier = min(min_barrier, barrier_area(child))
                won = won or int(child.levels_completed) > base
            direct, child_reach = fast_reach(child)
            item = (
                phase, pos, tuple(path + [5]), helper_row(child), minimum,
                min_barrier, len(child_reach), direct, won,
            )
            rows.append(item)
            print("case", item, flush=True)
        wait.append(2 if phase % 2 == 0 else 1)
    print(
        "best",
        sorted(
            rows,
            key=lambda x: (
                not x[8], x[4] is None,
                99 if x[4] is None else x[4],
                -x[6], x[5], len(x[2]),
            ),
        )[:10],
        flush=True,
    )


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
