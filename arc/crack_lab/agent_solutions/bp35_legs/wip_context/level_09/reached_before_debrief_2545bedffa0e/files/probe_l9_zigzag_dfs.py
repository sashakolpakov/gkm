"""Depth-first lookahead over left/up zigzags after the early wall flip."""

import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing
from probe_l9_route_variants import build_variant


def above_catch(env):
    avatars = connected_components(env.frame(), colors=(9, 11), min_area=3)
    if not avatars:
        return None
    avatar = avatars[0]
    ay0, ax0, _, ax1 = avatar.bbox
    candidates = []
    for blob in connected_components(env.frame(), colors=(15,), min_area=3):
        y0, x0, y1, x1 = blob.bbox
        if blob.area == 21 and y1 < ay0 and x0 <= ax1 and x1 >= ax0:
            candidates.append(blob)
    if not candidates:
        return None
    blob = max(candidates, key=lambda item: item.bbox[2])
    return 6, round(blob.centroid[1]), round(blob.centroid[0])


def make_root(env, lane):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    routed = build_variant(env, lane)
    for _ in range(5):
        routed.step(6, 57, 35)
    routed.step(*controls(routed)[0])
    return routed


def search(root, target=8, max_nodes=500, max_seconds=35):
    started = time.monotonic()
    stack = [(root.clone(), ())]
    best = ((), root.clone())
    nodes = 0
    while stack and nodes < max_nodes and time.monotonic() - started < max_seconds:
        node, path = stack.pop()
        nodes += 1
        if node.terminal():
            continue
        if len(path) > len(best[0]):
            best = path, node.clone()
        if int(node.levels_completed) >= 9 or len(path) >= target:
            return path, node, nodes
        up = above_catch(node)
        actions = [("L", (3,))]
        if up is not None:
            actions.append(("U", up))
        for token, action in reversed(actions):
            child = node.clone()
            child.step(*action)
            stack.append((child, path + ((token, action),)))
    return best[0], best[1], nodes


def probe(env):
    for lane in (6, 8):
        root = make_root(env.clone(), lane)
        path, final, nodes = search(root)
        print(
            "ZIGZAG",
            lane,
            "nodes",
            nodes,
            "depth",
            len(path),
            "path",
            path,
            "terminal",
            bool(final.terminal()),
            "level",
            int(final.levels_completed) + 1,
            "final",
            compact(final),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
