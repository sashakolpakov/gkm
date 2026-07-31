import json
import sys
from collections import defaultdict, deque

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


def brief(env):
    reward_path, reach = fast_reach(env)
    blobs = connected_components(
        env.frame(), colors=(11, 14, 15), min_area=1)
    areas = tuple(
        (color, sum(b.area for b in blobs if b.color == color))
        for color in (11, 15)
    )
    fronts = tuple((p, len(w))
                   for p, w in _special_frontier(reach, env.frame()))
    return int(env.levels_completed), len(reach), reward_path, auto_row(env), areas, fronts


def outcome(env, path):
    node = clone_after(env, path + [5])
    best_row = auto_row(node)
    best_tick = 0
    for tick in range(1, 31):
        node.step(2 if tick % 2 else 1)
        row = auto_row(node)
        if row is not None and (best_row is None or row < best_row):
            best_row, best_tick = row, tick
        if int(node.levels_completed) > 6:
            return (1, -1, tick, path + [5])
    return (0, 99 if best_row is None else best_row,
            best_tick, path + [5])


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    _, reach = fast_reach(env)
    targets = {pos for pos, _ in _special_frontier(reach, env.frame())}
    paths, states = arrivals(env, targets)
    groups = defaultdict(list)
    for path in paths:
        child = clone_after(env, path + [5])
        groups[key(child)].append(path + [5])
    print("groups", states, len(paths), len(groups))
    representatives = []
    for index, candidates in enumerate(groups.values()):
        candidates.sort(key=len)
        macro = candidates[0]
        child = clone_after(env, macro)
        print(index, macro, "histories", len(candidates), brief(child))
        representatives.append((child, macro))

    for index, (child, first_macro) in enumerate(representatives):
        _, child_reach = fast_reach(child)
        targets = {
            pos for pos, _ in _special_frontier(
                child_reach, child.frame())
        }
        second_paths, second_states = arrivals(
            child, targets, max_depth=14, max_states=1000)
        outcomes = sorted(
            (outcome(child, path) for path in second_paths),
            key=lambda item: (-item[0], item[1], len(item[3]), item[2]),
        )
        print("second", index, "states", second_states,
              "arrivals", len(second_paths),
              "best", None if not outcomes else outcomes[0],
              "combined", None if not outcomes else
              first_macro + outcomes[0][3])


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
