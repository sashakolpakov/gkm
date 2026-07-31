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


def visible_key(env):
    frame = np.asarray(env.frame())
    return np.where(np.isin(frame, (8, 9, 11, 14, 15)), frame, 0).tobytes()


def auto_row(env):
    blob = next(
        (b for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    return 99 if blob is None else blob.bbox[0]


def arrivals(env, targets, max_depth=14, max_states=400):
    def reconstruct(path):
        node = env.clone()
        for action in path:
            node.step(action)
        return node

    q = deque([[]])
    seen = {visible_key(env)}
    found = []
    while q and len(seen) <= max_states:
        path = q.popleft()
        node = reconstruct(path)
        if _avatar_pos(node.frame()) in targets:
            found.append(path)
        if len(path) >= max_depth or node.terminal():
            continue
        for action in MOVES:
            child_path = path + [action]
            child = reconstruct(child_path)
            state_key = visible_key(child)
            if state_key in seen:
                continue
            seen.add(state_key)
            q.append(child_path)
    return found


def search(env, max_stages=10, beam=30, histories=3):
    def reconstruct(path):
        node = env.clone()
        for action in path:
            node.step(action)
        return node

    base = int(env.levels_completed)
    frontier = [[]]
    for stage in range(max_stages + 1):
        groups = defaultdict(list)
        stats = []
        for prefix in frontier:
            node = reconstruct(prefix)
            reward_path, reach = fast_reach(node)
            if reward_path is not None:
                return prefix + reward_path
            fronts = _special_frontier(reach, node.frame())
            targets = {pos for pos, _ in fronts}
            stats.append((len(reach), len(fronts), auto_row(node)))
            if stage >= max_stages or not targets:
                continue
            for walk in arrivals(node, targets):
                arrived = reconstruct(prefix + walk)
                if int(arrived.levels_completed) > base:
                    return prefix + walk
                combined = prefix + walk + [5]
                child = reconstruct(combined)
                if int(child.levels_completed) > base:
                    return combined
                groups[visible_key(child)].append(combined)
        print("stage", stage, "frontier", len(frontier),
              "groups", len(groups), "stats", sorted(set(stats))[:20],
              flush=True)
        ranked = []
        for candidates in groups.values():
            candidates.sort(key=len)
            for prefix in candidates[:histories]:
                node = reconstruct(prefix)
                ranked.append((auto_row(node), len(prefix), prefix))
        ranked.sort(key=lambda item: (item[0], item[1]))
        frontier = [prefix for _, _, prefix in ranked[:beam]]
    return None


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    plan = search(env)
    print("plan", plan)
    node = clone_after(env, plan or [])
    print("result", int(node.levels_completed), auto_row(node))


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
