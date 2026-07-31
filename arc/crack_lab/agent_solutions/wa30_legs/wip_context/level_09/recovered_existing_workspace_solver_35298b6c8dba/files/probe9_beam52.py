"""Beam-search a <=52-turn finish after the verified remote-depot prefix."""

from collections import defaultdict

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import reach_level_9
from probe9_right_depot import FULL_TARGET, full_target_state
from probe9_verify import boxes


REMOTE = (
    [2] + [4] * 6 + [1, 5] + [1] * 2 + [5, 2]
    + [3] * 2 + [1, 5] + [4, 1, 5, 2]
    + [3] * 2 + [1] * 3 + [5] + [1] * 2 + [5, 2]
)


def cell(box):
    row0, col0, row1, col1 = box
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def features(env):
    empty, filled, occupied = full_target_state(env.frame())
    avatar_boxes = boxes(env.frame(), 14)
    avatar = cell(avatar_boxes[0]) if avatar_boxes else (-20, -20)
    holding = int((arr(env.frame()) == 0).sum()) > 4
    cargo_cells = tuple(cell(box) for box in boxes(env.frame(), 4))
    external = tuple(point for point in cargo_cells if point not in FULL_TARGET)
    destinations = empty if holding else external
    distance = min(
        (abs(avatar[0] - row) + abs(avatar[1] - col)
         for row, col in destinations),
        default=30,
    )
    thief_gone = not boxes(env.frame(), 15)
    score = (
        len(filled) * 1000
        + len(occupied) * 100
        + thief_gone * 180
        + holding * 90
        - distance
    )
    bucket = (len(filled), thief_gone, holding, avatar)
    return score, bucket, {
        "filled": len(filled),
        "occupied": len(occupied),
        "thief_gone": thief_gone,
        "holding": holding,
        "avatar": avatar,
        "distance": distance,
    }


def inspect(env):
    reach_level_9(env)
    start = env.clone()
    for action in REMOTE:
        start.step(action)
    base_level = start.levels_completed
    frontier = [(start, [])]
    transitions = 0
    width = 120
    for depth in range(1, 22):
        unique = {}
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                transitions += 1
                child_path = path + [action]
                if child.levels_completed > base_level:
                    print(
                        "BEAM52_WIN",
                        len(REMOTE) + depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return
                if not child.terminal():
                    unique.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
        ranked = sorted(
            unique.values(),
            key=lambda item: features(item[0])[0],
            reverse=True,
        )
        buckets = defaultdict(int)
        frontier = []
        for item in ranked:
            _, bucket, _ = features(item[0])
            if buckets[bucket] >= 3:
                continue
            buckets[bucket] += 1
            frontier.append(item)
            if len(frontier) >= width:
                break
        best_score, _, best_features = features(frontier[0][0])
        print(
            "BEAM52_DEPTH",
            depth,
            len(unique),
            transitions,
            best_score,
            best_features,
            frontier[0][1],
            flush=True,
        )
    print("BEAM52_RESULT", None, transitions, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
