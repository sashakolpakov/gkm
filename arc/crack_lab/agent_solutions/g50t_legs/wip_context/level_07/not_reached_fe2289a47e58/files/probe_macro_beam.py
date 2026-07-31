import json
import sys
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import MOVES, _avatar_pos
from perception import connected_components


PREFIX = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
    + [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]
)


def visible_key(env):
    return np.asarray(env.frame()).tobytes()


def helper_row(env):
    blob = next(
        (b for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    return 99 if blob is None else blob.bbox[0]


def rank(env, path):
    blobs = connected_components(
        env.frame(), colors=(11, 14, 15), min_area=1
    )
    barrier = sum(b.area for b in blobs if b.color == 15)
    switch = sum(b.area for b in blobs if b.color == 11)
    return (helper_row(env), barrier, -switch, len(path))


def movement_commits(root, base, max_depth=18, max_states=420):
    queue = deque([(root.clone(), [], False)])
    seen = {visible_key(root): 1}
    expanded = 0
    children = defaultdict(list)
    best_row = helper_row(root)
    while queue and expanded < max_states:
        node, path, at_switch = queue.popleft()
        expanded += 1
        if int(node.levels_completed) > base:
            return path, [], len(seen), best_row
        best_row = min(best_row, helper_row(node))
        if at_switch:
            child = node.clone()
            child.step(5)
            combined = path + [5]
            if int(child.levels_completed) > base:
                return combined, [], len(seen), best_row
            if not child.terminal() and helper_row(child) != 99:
                group = children[visible_key(child)]
                if len(group) < 2:
                    group.append((child, combined))
        if len(path) >= max_depth or node.terminal():
            continue
        before_pos = _avatar_pos(node.frame())
        before_frame = np.asarray(node.frame())
        for action in MOVES:
            child = node.clone()
            child.step(action)
            if int(child.levels_completed) > base:
                return path + [action], [], len(seen), best_row
            if child.terminal() or _avatar_pos(child.frame()) == before_pos:
                continue
            state_key = visible_key(child)
            if seen.get(state_key, 0) >= 2:
                continue
            seen[state_key] = seen.get(state_key, 0) + 1
            new_pos = _avatar_pos(child.frame())
            r, c = new_pos
            footprint = before_frame[r:r + 5, c:c + 5]
            landed = bool(np.any(~np.isin(footprint, (0, 1, 2, 5, 9))))
            queue.append((child, path + [action], landed))
    flat = [item for group in children.values() for item in group]
    return None, flat, len(seen), best_row


def search(root, max_stages=12, beam=12, histories=2):
    base = int(root.levels_completed)
    frontier = [(root.clone(), [])]
    for stage in range(max_stages):
        groups = defaultdict(list)
        stats = []
        for node, prefix in frontier:
            win, children, states, best_row = movement_commits(node, base)
            stats.append((states, best_row, len(children)))
            if win is not None:
                return prefix + win
            for child, macro in children:
                combined = prefix + macro
                group = groups[visible_key(child)]
                if len(group) < histories:
                    group.append((child, combined))
        ranked = [
            (rank(child, path), child, path)
            for group in groups.values()
            for child, path in group
        ]
        ranked.sort(key=lambda item: item[0])
        frontier = [(child, path) for _, child, path in ranked[:beam]]
        print(
            "stage", stage + 1,
            "parents", len(stats),
            "groups", len(groups),
            "frontier", len(frontier),
            "rows", sorted({helper_row(node) for node, _ in frontier}),
            "stats", stats[:8],
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
    plan = search(env)
    print("plan", plan, flush=True)
    if plan:
        for action in plan:
            env.step(action)
            if env.terminal():
                break
    print("end", int(env.levels_completed), len(plan or []), flush=True)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
