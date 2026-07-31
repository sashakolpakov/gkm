"""Find and inspect the nearest level-2 contact using observational clone BFS."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


def cells(frame, colors):
    a = perception.arr(frame)
    return {
        (r, c)
        for r in range(2, 62, 3)
        for c in range(2, 62, 3)
        if int(a[r + 1, c + 1]) in colors
    }


def gap(frame):
    ours = cells(frame, {0})
    others = cells(frame, {9, 11, 14})
    return min(abs(r - s) + abs(c - t) for r, c in ours for s, t in others) // 3 - 1


def probe(env):
    play_level_1(env)
    path = perception.bounded_bfs(
        env,
        lambda node, _: gap(node.frame()) <= 0,
        actions=(1, 2, 3, 4, 5),
        max_states=6000,
        max_depth=35,
    )
    print("contact_path", path, "len", None if path is None else len(path))
    if path is None:
        return
    for length in range(len(path) + 1):
        prefix = perception.replay(env, path[:length])
        black = cells(prefix.frame(), {0})
        pegs = sorted(cells(prefix.frame(), {8}))
        print(
            "prefix",
            length,
            "action",
            None if length == 0 else path[length - 1],
            "top",
            perception.color_counts(perception.arr(prefix.frame())[0]),
            "peg_gaps",
            sorted(
                (min(abs(r - s) + abs(c - t) for s, t in black) // 3, (r // 3, c // 3))
                for r, c in pegs
            )[:5],
        )
    node = perception.replay(env, path)
    print("contact_gap", gap(node.frame()), "colors", perception.color_counts(node.frame()))
    print(
        "contact_objects",
        [
            (b.color, b.bbox, b.area)
            for b in perception.connected_components(
                node.frame(), colors=(0, 9, 11, 14), min_area=4
            )
        ],
    )
    base = perception.arr(node.frame())
    for action in env.actions:
        child = node.clone()
        child.step(action)
        changed_fixed = int(
            np.count_nonzero(
                np.isin(base, (9, 11, 14))
                != np.isin(perception.arr(child.frame()), (9, 11, 14))
            )
        )
        print(
            "after",
            action,
            "level",
            child.levels_completed,
            "gap",
            gap(child.frame()),
            "fixed_delta",
            changed_fixed,
            "delta",
            perception.frame_delta(base, child.frame())["count"],
            "top",
            perception.color_counts(perception.arr(child.frame())[0]),
        )
        print(
            "objects",
            [
                (b.color, b.bbox, b.area)
                for b in perception.connected_components(
                    child.frame(), colors=(0, 9, 11, 14), min_area=4
                )
            ],
        )


arena.run_program("cn04", probe)
