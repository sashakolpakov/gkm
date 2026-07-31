import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
N1 = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
TARGET9 = ((28, 14), (28, 20), (28, 50))


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def click(env, mode):
    color = 6 if mode == "h" else 15
    blobs = [
        blob
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=16
        )
        if blob.centroid[0] < 53
    ]
    blob = min(
        blobs,
        key=lambda item: item.centroid[1] if mode == "h"
        else item.centroid[0],
    )
    return (6, round(blob.centroid[1]), round(blob.centroid[0]))


def apply(env, path):
    for action in path:
        if action in ("h", "v"):
            env.step(*click(env, action))
        elif isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)


def positions(env, color):
    return tuple(sorted(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=12
        )
        if blob.centroid[0] < 53
    ))


def root_state(env):
    full8 = (
        E1 + ["v", 2, 1, "h"] + EN + ["v", 2, "h", 3, 3, 3]
        + ["h"] + EN + ["v", 2, "h", 3, 3, 3]
    )
    park = (
        full8 + ["v", 1, 1, 1, "h"]
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5
        + ["h", 2, 2, 2, 4, "v", 2] + N1
    )
    n2_stage = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4
    n2_fix_release = [
        "h", 3, "v", 2, "h", 4, "v", 1, 1, 1
    ]
    node = env.clone()
    apply(node, park + n2_stage + n2_fix_release)
    return node


def ownership_moves(env):
    if positions(env, 9) != TARGET9:
        return 9
    node = env.clone()
    apply(node, ["h", 1])
    before = set(TARGET9)
    after = set(positions(node, 9))
    return len(before - after) + len(after - before)


def search(root, max_states=2000, max_depth=10):
    serial = itertools.count()
    start_value = ownership_moves(root)
    heap = [(start_value, 0, next(serial), root, "v", ())]
    seen = {np.asarray(root.frame()).tobytes()}
    best = (start_value, (), positions(root, 9))
    expanded = 0
    while heap and expanded < max_states:
        _, depth, _, node, mode, path = heapq.heappop(heap)
        expanded += 1
        value = ownership_moves(node)
        if value < best[0]:
            best = (value, path, positions(node, 9))
        if value == 0:
            return list(path), expanded, best
        if depth >= max_depth:
            continue
        for option in (1, 2, 3, 4, "switch"):
            child = node.clone()
            next_mode = mode
            if option == "switch":
                next_mode = "h" if mode == "v" else "v"
                action = click(child, next_mode)
                child.step(*action)
            else:
                action = option
                child.step(action)
            key = np.asarray(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            next_path = path + (action,)
            score = depth + 1 + 3 * ownership_moves(child)
            heapq.heappush(
                heap,
                (score, depth + 1, next(serial), child, next_mode, next_path),
            )
    return None, expanded, best


def probe(env):
    reach_level_6(env)
    root = root_state(env)
    print("ROOT", positions(root, 9), "OWN", ownership_moves(root))
    path, expanded, best = search(root)
    print("FOUND", path, "EXPANDED", expanded, "BEST", best)


A.run_program("sk48", probe)
