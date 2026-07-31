import json
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import MOVES, _avatar_pos, _special_frontier, clone_after, fast_reach
from perception import connected_components


PREFIX = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
    + [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]
)


def auto_row(env):
    blob = next(
        (b for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    return None if blob is None else blob.bbox[0]


def key(env):
    frame = np.asarray(env.frame())
    return np.where(np.isin(frame, (8, 9, 11, 14, 15)), frame, 0).tobytes()


def arrivals(env, targets, max_depth=18, max_states=4000):
    q = deque([(env.clone(), [])])
    seen = {key(env)}
    found = []
    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if _avatar_pos(node.frame()) in targets:
            found.append(path)
        if len(path) >= max_depth:
            continue
        for action in MOVES:
            child = node.clone()
            child.step(action)
            state_key = key(child)
            if state_key in seen:
                continue
            seen.add(state_key)
            q.append((child, path + [action]))
    return found, len(seen)


def outcome(env, path):
    node = clone_after(env, path + [5])
    best_row = auto_row(node)
    best_tick = 0
    for tick in range(1, 41):
        node.step(2 if tick % 2 else 1)
        row = auto_row(node)
        if row is not None and (best_row is None or row < best_row):
            best_row, best_tick = row, tick
        if int(node.levels_completed) > 6:
            return (1, -1, tick, path)
    return (0, 99 if best_row is None else best_row, best_tick, path)


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    reward_path, reach = fast_reach(env)
    fronts = _special_frontier(reach, env.frame())
    targets = {pos for pos, _ in fronts}
    print("start", len(reach), reward_path,
          [(p, len(w)) for p, w in fronts], auto_row(env))
    paths, states = arrivals(env, targets)
    results = sorted(
        (outcome(env, path) for path in paths),
        key=lambda item: (-item[0], item[1], len(item[3]), item[2]),
    )
    print("search", states, "arrivals", len(paths))
    for item in results[:30]:
        print("candidate", item)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
