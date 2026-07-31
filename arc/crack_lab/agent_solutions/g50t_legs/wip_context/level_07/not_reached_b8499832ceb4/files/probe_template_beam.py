import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _avatar_pos, _special_frontier, fast_reach
from perception import connected_components


PREFIX = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
    + [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]
)

ARRIVALS = (
    [2, 3],
    [2, 3, 2, 1],
    [1, 2, 1, 2, 2, 3],
    [2, 3, 2, 1, 2, 1, 2, 1],
    [1, 2, 1, 2, 2, 3, 4, 1, 2, 3],
    [1, 2, 1, 2, 2, 3, 4, 1, 1, 3, 3, 1, 1],
    [1, 2, 1, 2, 2, 3, 4, 1, 1, 3, 4, 2, 2, 3],
    [1, 2, 1, 2, 2, 3, 4, 1, 1, 3, 3, 1, 2, 4, 3, 1, 1],
    [1, 2, 1, 2, 2, 3, 4, 1, 1, 3, 3, 1, 2, 4, 4, 2, 2, 3],
)


def helper_row(env):
    try:
        blobs = connected_components(
            env.frame(), colors=(14,), min_area=4
        )
    except (IndexError, ValueError):
        return 99
    blob = next(iter(blobs), None)
    return 99 if blob is None else blob.bbox[0]


def visual_key(env):
    return np.asarray(env.frame()).tobytes()


def rank(env, path):
    blobs = connected_components(
        env.frame(), colors=(11, 15), min_area=1
    )
    barrier = sum(b.area for b in blobs if b.color == 15)
    switches = sum(b.area for b in blobs if b.color == 11)
    return (helper_row(env), barrier, -switches, len(path))


def replay(root, path):
    node = root.clone()
    for action in path:
        if node.terminal():
            break
        try:
            node.step(action)
        except (IndexError, ValueError):
            return None
    return node


def apply_macro(node, macro):
    try:
        child = node.clone()
    except (IndexError, ValueError):
        return None
    for action in macro:
        try:
            if child.terminal():
                break
            child.step(action)
        except (IndexError, ValueError):
            return None
    return child


def search(root, max_stages=16, beam=100, histories=5):
    base = int(root.levels_completed)
    frontier = [(root.clone(), [])]
    for stage in range(max_stages):
        groups = defaultdict(list)
        arrivals = ARRIVALS if stage == 0 else tuple([2] + p for p in ARRIVALS)
        for node, prefix in frontier:
            for arrival in arrivals:
                combined = prefix + arrival + [5]
                child = apply_macro(node, arrival + [5])
                if child is None:
                    continue
                try:
                    if int(child.levels_completed) > base:
                        return combined
                    child_key = visual_key(child)
                except (IndexError, ValueError):
                    continue
                group = groups[child_key]
                if len(group) < histories:
                    group.append((child, combined))
        ranked = []
        for group in groups.values():
            for child, path in group:
                try:
                    item_rank = rank(child, path)
                except (IndexError, ValueError):
                    continue
                ranked.append((item_rank, child, path))
        ranked.sort(key=lambda item: item[0])
        frontier = [(child, path) for _, child, path in ranked[:beam]]
        best_path = None if not ranked else ranked[0][2]
        best_fronts = ()
        if best_path is not None:
            best_node = ranked[0][1]
            _, best_reach = fast_reach(best_node)
            best_fronts = tuple(
                (pos, len(walk))
                for pos, walk in _special_frontier(
                    best_reach, best_node.frame()
                )
            )
        print(
            "stage", stage + 1,
            "groups", len(groups),
            "frontier", len(frontier),
            "rows", sorted({helper_row(node) for node, _ in frontier}),
            "best", None if not ranked else ranked[0][0],
            "fronts", best_fronts,
            "path", best_path,
            flush=True,
        )
        if not frontier:
            break
    return None


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    print("root", _avatar_pos(env.frame()), helper_row(env), flush=True)
    plan = search(env)
    print("plan", plan, flush=True)
    child = replay(env, plan or [])
    print("end", int(child.levels_completed), helper_row(child), flush=True)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
