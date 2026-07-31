"""Bounded local BFS over verified post-flip affordances."""

import sys
import time
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing
from probe_l9_route_variants import build_variant


def avatar_blob(env):
    blobs = [
        blob
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    return blobs[0] if blobs else None


def local_actions(env):
    avatar = avatar_blob(env)
    if avatar is None:
        return ()
    ay, ax = avatar.centroid
    actions = [(3,), (4,)]
    for blob in connected_components(
        env.frame(), colors=(8, 12, 14, 15), min_area=3
    ):
        if blob.bbox[0] >= 63:
            continue
        by, bx = blob.centroid
        local = abs(by - ay) <= 8 and abs(bx - ax) <= 8
        remote = blob.color in (8, 14) and blob.area >= 21
        if local or remote:
            actions.append((6, round(bx), round(by)))
    return tuple(dict.fromkeys(actions))


def frame_key(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed),
        arr(env.frame()).tobytes(),
    )


def postflip_root(env, lane=6):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    routed = build_variant(env, lane)
    for _ in range(5):
        routed.step(6, 57, 35)
    routed.step(*controls(routed)[0])
    return routed


def search(root, target_depth=8, max_states=500, max_seconds=30):
    started = time.monotonic()
    queue = deque([(root.clone(), ())])
    seen = {(0, frame_key(root))}
    best = ((), root.clone())
    while queue and len(seen) <= max_states:
        if time.monotonic() - started > max_seconds:
            break
        node, path = queue.popleft()
        if len(path) > len(best[0]) and not node.terminal():
            best = path, node.clone()
        if int(node.levels_completed) >= 9:
            return "reward", path, node, len(seen)
        if len(path) >= target_depth or node.terminal():
            if len(path) >= target_depth and not node.terminal():
                return "survival", path, node, len(seen)
            continue
        for action in local_actions(node):
            child = node.clone()
            child.step(*action)
            key = (len(path) + 1, frame_key(child))
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + (action,)))
    return "best", best[0], best[1], len(seen)


def probe(env):
    for lane in (6,):
        root = postflip_root(env.clone(), lane)
        status, path, final, states = search(root)
        print(
            "SEARCH",
            lane,
            status,
            "states",
            states,
            "depth",
            len(path),
            "path",
            path,
            "terminal",
            bool(final.terminal()),
            "final",
            compact(final),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
