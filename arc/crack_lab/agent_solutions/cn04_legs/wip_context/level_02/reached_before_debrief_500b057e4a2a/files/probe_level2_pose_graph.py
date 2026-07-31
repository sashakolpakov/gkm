"""Search peg-visit histories cheaply on each mover's finite pose graph."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


SOURCE = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11)}
TARGET = {"A": (0, 2, 4, 5), "B": (6, 7, 8, 11),
          "C": (9, 10)}
TARGET_COLOR = {"A": 14, "B": 11, "C": 9}
SELECT_COLOR = {"B": 14, "C": 11}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def bridged(node, peg, target_color):
    a = perception.arr(node.frame())
    r, c = peg
    if int(a[r, c]) != 8:
        return False
    return any(
        {first, second} == {0, target_color}
        for first, second in (
            (int(a[r - 3, c]), int(a[r + 3, c])),
            (int(a[r, c - 3]), int(a[r, c + 3])),
        )
    )


def pose_graph(root, pegs, target, target_color):
    nodes = [root.clone()]
    paths = [[]]
    keys = {avatar_key(root): 0}
    edges = []
    masks = []
    docks = []
    cursor = 0
    while cursor < len(nodes) and len(nodes) < 4000:
        node = nodes[cursor]
        masks.append(sum(1 << i for i, peg in enumerate(pegs)
                         if covered(node, peg)))
        docks.append(sum(1 << i for i in target
                         if bridged(node, pegs[i], target_color)))
        row = []
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in keys:
                keys[key] = len(nodes)
                nodes.append(child)
                paths.append(paths[cursor] + [action])
            row.append(keys[key])
        edges.append(tuple(row))
        cursor += 1
    return edges, masks, docks


def mask_search(edges, masks, docks, required, target):
    wanted = sum(1 << i for i in required)
    q = deque([(0, masks[0], [])])
    seen = {(0, masks[0] & wanted)}
    while q:
        pose, mask, path = q.popleft()
        if mask & wanted == wanted and (docks[pose] & sum(1 << i for i in target)).bit_count() >= 2:
            return path, len(seen)
        for offset, child in enumerate(edges[pose]):
            child_mask = mask | masks[child]
            key = (child, child_mask & wanted)
            if key not in seen:
                seen.add(key)
                q.append((child, child_mask, path + [offset + 1]))
    return None, len(seen)


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for name in "ABC":
        node = env.clone()
        if name in SELECT_COLOR:
            select_color(node, SELECT_COLOR[name])
        edges, masks, docks = pose_graph(
            node, pegs, TARGET[name], TARGET_COLOR[name]
        )
        for label, required in (
            ("target", TARGET[name]),
            ("both", SOURCE[name] + TARGET[name]),
        ):
            path, states = mask_search(
                edges, masks, docks, required, TARGET[name]
            )
            print("agent", name, label, "poses", len(edges),
                  "states", states, "path", path)


arena.run_program("cn04", probe)
